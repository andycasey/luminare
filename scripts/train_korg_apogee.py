"""Train a model."""

import sys
import gc
import jax
import jax.numpy as jnp
import pickle
import numpy as np
import h5py as h5
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from luminare import nmf, fourier
from luminare.utils import parse_marcs_photosphere_path

input_path = sys.argv[1]
output_path = sys.argv[2]

"""
Notes:

- On an A100 with:
    - pixel_subsampling = 4
    - n_modes = (5, 5, 5, 4, 7, 11, 15)
    - n_components = 64
    - wh_iterations = 1000

    It solves very quickly (a few minutes).

    Doing without pixel subsampling requires ~4 GPUs.

- On an A100 with:
    - pixel_subsampling = 4
    - n_modes = (6, 6, 5, 4, 7, 11, 15) 
    - n_components = 128
    - wh_iterations = 10_000

    It takes about 15 minutes to run, and produces a very good model.
"""

n_components = 128
wh_iterations = 10_000
h_iterations = 1_000
epsilon = 1e-12
# n_grid_points: (6, 6, 5, 4, 8, 12, 17)
n_modes = (6, 6, 5, 4, 7, 11, 15) 

seed = 42
bit_precision = 32
verbose_frequency = 100
pixel_subsampling = 3

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

        parameter_names = ("N_m", "C_m", "alpha_m", "vmic", "m_H", "logg", "Teff")
        slices = (
            slice(0, None), # N_m
            slice(0, None), # C_m
            slice(0, None), # alpha_m
            slice(0, None), # vmic
            slice(0, None), # m_H
            slice(0, None), # logg
            slice(0, 17),   # Teff (keep above 4000 K to maintain rectilinear)
        )
        flip_axes = (parameter_names.index("Teff"), )  # Flip the Teff axis

        min_parameters = np.array([np.min(fp[k][s]) for k, s in zip(parameter_names, slices)])
        max_parameters = np.array([np.max(fp[k][s]) for k, s in zip(parameter_names, slices)])
        grid_parameters = [fp[k][s] for k, s in zip(parameter_names, slices)]    
        absorption = fp["spectra"][slices]
        mask = (
            (absorption > 0).all(axis=-1)
        *   (fp["converged_flag"][slices] == 1)
        )
        teff_index = parameter_names.index("Teff")
        n_removed_by_holes, skipped_out_of_bounds, skipped_not_vmic = (0, 0, 0)
        with open("marcs_filled_atmosphere_paths.txt", "r") as marcs_fp:
            for line in marcs_fp:
                r = parse_marcs_photosphere_path(line)
                if not (max_parameters[teff_index] >= r["Teff"] >= min_parameters[teff_index]):
                    skipped_out_of_bounds += 1
                    continue

                is_plane_parallel = (r["logg"] >= 3.5)
                is_spherical = not is_plane_parallel
                if (
                    (is_plane_parallel and r["vmic"] == 1.0)
                |   (is_spherical and r["vmic"] == 2.0)
                ):
                    # all things in that v_micro slice
                    indices = [slice(None)] * len(parameter_names)
                    for i, pn in enumerate(parameter_names):
                        if pn in ("vmic", "N_m"): continue
                        indices[i] = np.where(fp[pn][:] == r[pn])[0]
                        
                    indices = tuple(indices)
                    mask[indices] = False
                    n_removed_by_holes += mask[indices].size
                else:
                    skipped_not_vmic += 1    

        if flip_axes is not None:
            for axes in flip_axes:
                grid_parameters[axes] = np.flip(grid_parameters[axes])

            mask = np.flip(mask, axis=flip_axes)
            absorption = np.flip(absorption, axis=flip_axes)

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