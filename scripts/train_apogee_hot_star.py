"""Train a model on APOGEE model spectra of hot stars."""

import sys
import gc
import jax
import jax.numpy as jnp
import pickle
import numpy as np
import h5py as h5
from jax.sharding import Mesh, PartitionSpec, NamedSharding

sys.path.insert(0, "../src")
from luminare import nmf, fourier

input_path = "../data/hot_stars/apogee/hot_stars_apogee.h5"
output_path = "hot_stars_apogee.model"


n_components = 8
wh_iterations = 100_000
h_iterations = 10_000
epsilon = 1e-12
n_modes = (31, 21, 9)

seed = 42
bit_precision = 32
verbose_frequency = 100
pixel_subsampling = None

meta = dict(
    seed=seed,
    h_iterations=h_iterations,
    wh_iterations=wh_iterations,
    n_components=n_components,
    epsilon=epsilon,
    bit_precision=bit_precision,
)

jax.config.update("jax_enable_x64", bit_precision > 32)
jax.config.update("jax_default_matmul_precision", "float32")

n_gpus = jax.device_count("gpu")
print(f"Number of GPUs detected: {n_gpus}")
default_mesh = jax.make_mesh((n_gpus,), axis_names=("s",))
default_sharding = NamedSharding(default_mesh, PartitionSpec("s", None))
replicated_sharding = NamedSharding(default_mesh, PartitionSpec())

try:
    absorption
except NameError:        
    with h5.File(input_path, "r") as fp:    
        parameter_names = tuple(map(str, fp.attrs["parameter_names"]))
        slices = (
            slice(30, None), # Teff: keep above 10_000 K initially
            slice(0, None), # logg
            slice(0, None), # m_H
        )

        min_parameters = np.array([np.min(fp[k][s]) for k, s in zip(parameter_names, slices)])
        max_parameters = np.array([np.max(fp[k][s]) for k, s in zip(parameter_names, slices)])
        grid_parameters = [fp[k][s] for k, s in zip(parameter_names, slices)]    
        absorption = fp["flux"][slices]
        
        print(f"Absorption shape: {absorption.shape}")

        mask = (
           (fp["converged_flag"][slices] == 1)
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
        n_grid_points = tuple(absorption.shape[:-1])
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

HVTA = fourier.rmatmat_jit(n_grid_points, n_modes, V @ H.T)
ATA = fourier.gram_diagonal_jit(n_grid_points, n_modes, M)

HVTA_ATA_inv = HVTA / ATA.reshape((1, -1))

print("Computing Cholesky factorization of H @ H.T")
cho_factor = jax.scipy.linalg.cho_factor(H @ H.T)

print("Solving for X")
X = jax.vmap(jax.scipy.linalg.cho_solve, in_axes=(None, 1))(cho_factor, HVTA_ATA_inv)

print("Compute A @ X")
W = fourier.matmat_jit(n_grid_points, n_modes, X).T
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
    H=np.array(H),
    X=np.array(X),
    parameter_names=parameter_names,
    min_parameters=np.array(min_parameters),
    max_parameters=np.array(max_parameters),
    n_modes=n_modes,
    meta=meta
)

with open(output_path, "wb") as fp:
    pickle.dump(serialised_model, fp)