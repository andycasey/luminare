"""Train a model on APOGEE model spectra of hot stars."""

import sys
import gc
import jax
import jax.numpy as jnp
import pickle
import numpy as np
import h5py as h5
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

sys.path.insert(0, "../src")
from luminare import nmf, fourier

input_path = "../data/hot_stars/apogee/hot_stars_apogee.h5"
output_path = "hot_stars_apogee.model"


n_components = 16
wh_iterations = 1_000
h_iterations = 10_000
epsilon = 1e-12
n_modes = (81, 21, 9)

seed = 42
bit_precision = 32
verbose_frequency = 100
pixel_subsampling = None
cholesky = True

meta = dict(
    seed=seed,
    h_iterations=h_iterations,
    wh_iterations=wh_iterations,
    n_components=n_components,
    epsilon=epsilon,
    bit_precision=bit_precision,
    cholesky=cholesky
)

jax.config.update("jax_enable_x64", bit_precision > 32)
jax.config.update("jax_default_matmul_precision", "highest" if bit_precision > 32 else "float32")

n_gpus = jax.device_count("gpu")
print(f"Number of GPUs detected: {n_gpus}")
default_mesh = jax.make_mesh((n_gpus,), axis_names=("s",))
default_sharding = NamedSharding(default_mesh, PartitionSpec("s", None))
replicated_sharding = NamedSharding(default_mesh, PartitionSpec())

try:
    absorption
except NameError:        
    with h5.File(input_path, "r") as fp:    
        λ = fp["wavelength"][:]
        parameter_names = tuple(map(str, fp.attrs["parameter_names"]))
        # The sampling from Teff: 7000 to 10_000 is in 100 K increments, but we 
        # want 250 K increments to keep it rectilinear. Here we will just interpolate.
        min_parameters = np.array([np.min(fp[k]) for k in parameter_names])
        max_parameters = np.array([np.max(fp[k]) for k in parameter_names])

        teff_index = parameter_names.index("Teff")
        Teff = np.arange(min_parameters[teff_index], max_parameters[teff_index]+1, 250)

        shape = (Teff.size, *tuple(fp["flux"].shape[1:]))
        absorption = np.nan * np.ones(shape, dtype=fp["flux"].dtype)
        converged = np.zeros(shape[:-1], dtype=fp["converged_flag"].dtype)
        for i, teff in enumerate(Teff):
            index = np.where(fp["Teff"][:] == teff)[0]
            if index.size == 0:
                continue
            absorption[i] = fp["flux"][index[0]]
            converged[i] = fp["converged_flag"][index[0]]
        
        interpolated_teffs = sorted(set(Teff).difference(set(fp["Teff"][:])))
        for teff in interpolated_teffs:
            i = Teff.searchsorted(teff)
            j = fp["Teff"][:].searchsorted(teff)
            # do something dumb
            absorption[i] = 0.5 * (fp["flux"][j-1] + fp["flux"][j])
            converged[i] = (fp["converged_flag"][j-1] & fp["converged_flag"][j])

        grid_parameters = [fp[k][:] for k in parameter_names]
        grid_parameters[teff_index] = Teff

        print(f"Absorption shape: {absorption.shape}")

        mask = (
            converged
        )

        r = mask.sum() % n_gpus
        if r > 0:
            print(f"Removing {r:,} grid points to make the mask divisible by {n_gpus}.")
            mask[np.where(mask)[0][-r:]] = False

        #absorption[~mask] = 0.0 # for missing data
        # absorption = 1 - absorption
        if pixel_subsampling is not None:
            absorption = absorption[..., ::pixel_subsampling] 
            print(f"ONLY KEEPING EVERY {pixel_subsampling}TH PIXEL")
    
        absorption *= -1
        absorption += 1
        n_points_per_parameter = tuple(absorption.shape[:-1])
        M = jnp.array(mask.astype(int))
else:
    print("Using existing absorption data.")

key = jax.random.PRNGKey(seed)

V = jax.device_put(absorption[mask], default_sharding)
V = jnp.clip(V, 0, None)

n_spectra, n_pixels = V.shape

H = jax.random.uniform(key, shape=(n_components, n_pixels), minval=epsilon, maxval=1)
W = jax.device_put(
    jax.random.uniform(key, shape=(n_spectra, n_components), minval=epsilon, maxval=1),
    default_sharding,
)

print(f"Shapes:")
print(f"\tV: [{V.shape[0]:,} x {V.shape[1]:,}]")
print(f"\tW: [{W.shape[0]:,} x {W.shape[1]:,}]")
print(f"\tH: [{H.shape[0]:,} x {H.shape[1]:,}]")

W, H, wh_losses = nmf.multiplicative_updates_WH(
    V,
    W,
    H,
    iterations=wh_iterations,
    verbose_frequency=verbose_frequency,
    epsilon=epsilon,
)

grid = np.array(np.meshgrid(*grid_parameters, indexing="ij")).reshape((len(parameter_names), -1)).T

nW = np.nan * np.ones((*shape[:-1], n_components), dtype=W.dtype)
nW[mask] = np.array(W)
nW = nW.reshape((-1, n_components))
grid[:, 0] = np.log10(grid[:, 0])

rf = RandomForestRegressor(
    n_estimators=500,          # More trees for better stability and variance reduction
    max_depth=5,               # Limit tree depth to prevent overfitting
    min_samples_split=30,      # Require more samples to make a split (reduces overfitting)
    min_samples_leaf=15,       # Require more samples in leaf nodes (smoother predictions)
    max_features='sqrt',       # Use sqrt(n_features) for each split (reduces overfitting)
    bootstrap=True,            # Use bootstrap sampling (default, but explicit)
    max_samples=0.8,           # Use 80% of samples for each tree (adds more diversity)
    oob_score=True,            # Compute out-of-bag score for validation
    n_jobs=-1, 
    random_state=seed
)
rf.fit(nW, grid)

pred = rf.predict(nW)

fig, axes = plt.subplots(len(parameter_names), 1, figsize=(6, 3*len(parameter_names)), squeeze=False)
for i, (name, ax) in enumerate(zip(parameter_names, axes.flat)):
    ok = np.isfinite(nW).all(axis=1)
    x, y = grid[ok, i], pred[ok, i]
    ax.scatter(x, y, s=1, alpha=0.5)

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))
    ax.plot(limits, limits, c="k", ls="--", alpha=0)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(name)
    title = f"mean: {np.mean(x-y):.2f}, std: {np.std(x-y):.2f}"
    ax.set_title(title)
    print(f"{name} -> {title}")
    
fig.tight_layout()
fig.savefig(f"{output_path}_rf_performance.png", dpi=300)

raise a

with open(f"{output_path}.rf", "wb") as fp:
    pickle.dump(rf, fp)

del W
gc.collect()
jax.clear_caches()

H = jnp.where(H > epsilon, H, 0.0)

H = jax.device_put(H, replicated_sharding)
M = jax.device_put(M, default_sharding)

try:
    V = jax.device_put(absorption.reshape((-1, n_pixels)), default_sharding)
except ValueError:
    print("Not sharding V, using default device.")
    V = jnp.array(absorption.reshape((-1, n_pixels)))

V = jnp.clip(jnp.nan_to_num(V, nan=0.0), 0, None)

# Set masked values to zero
V *= M.reshape((-1, 1))

H = jnp.clip(H, epsilon, None)

HVTA = fourier.rmatmat_jit(n_points_per_parameter, n_modes, V @ H.T)
ATA = fourier.gram_diagonal_jit(n_points_per_parameter, n_modes, M)

HVTA_ATA_inv = HVTA / ATA.reshape((1, -1))

if cholesky:
    print("Computing Cholesky factorization of H @ H.T")
    cho_factor = jax.scipy.linalg.cho_factor(H @ H.T)

    print("Solving for X")
    X = jax.vmap(jax.scipy.linalg.cho_solve, in_axes=(None, 1))(cho_factor, HVTA_ATA_inv)    
else:
    print("Solving for X")
    X = jax.vmap(jax.scipy.linalg.solve, in_axes=(None, 1))(H @ H.T, HVTA_ATA_inv)
    
print("Compute A @ X")
W = fourier.matmat_jit(n_points_per_parameter, n_modes, X).T
W = jnp.clip(W, epsilon, None)

print(f"Minimize C(H|X,V)")
H, losses = nmf.multiplicative_updates_H(
    V, W, H, 
    iterations=h_iterations, 
    verbose_frequency=verbose_frequency,
    epsilon=epsilon
)
H = jnp.where(H > epsilon, H, 0.0)

serialised_model = dict(
    λ=λ,
    H=np.array(H),
    X=np.array(X),
    parameter_names=parameter_names,
    min_parameters=min_parameters,
    max_parameters=max_parameters,
    n_modes=n_modes,
    n_points_per_parameter=n_points_per_parameter,
    spectral_resolution=800_000,
    medium="air",
    meta=meta
)

with open(output_path, "wb") as fp:
    pickle.dump(serialised_model, fp)