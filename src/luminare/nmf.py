import jax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm
from typing import Tuple, List, Optional


@jax.jit
def loss(W: jnp.ndarray, H: jnp.ndarray, V: jnp.ndarray) -> float:
    """
    Compute the Frobenius norm loss for NMF: ||WH - V||_F^2 / n_samples.

    Args:
        W: Factor matrix of shape (n_samples, n_components)
        H: Factor matrix of shape (n_components, n_features)
        V: Data matrix of shape (n_samples, n_features)

    Returns:
        Mean squared reconstruction error
    """
    return jnp.sum(jnp.square(W @ H - V)) / V.shape[0]


@jax.jit
def max_abs_diff(W: jnp.ndarray, H: jnp.ndarray, V: jnp.ndarray) -> float:
    """
    Compute the maximum absolute reconstruction error.

    Args:
        W: Factor matrix of shape (n_samples, n_components)
        H: Factor matrix of shape (n_components, n_features)
        V: Data matrix of shape (n_samples, n_features)

    Returns:
        Maximum absolute difference between WH and V
    """
    return jnp.max(jnp.abs(W @ H - V))


@jax.jit
def mean_abs_diff(W: jnp.ndarray, H: jnp.ndarray, V: jnp.ndarray) -> float:
    """
    Compute the mean absolute reconstruction error.

    Args:
        W: Factor matrix of shape (n_samples, n_components)
        H: Factor matrix of shape (n_components, n_features)
        V: Data matrix of shape (n_samples, n_features)

    Returns:
        Mean absolute difference between WH and V
    """
    return jnp.mean(jnp.abs(W @ H - V))


@partial(jax.jit, donate_argnums=(0,), static_argnames=("iterations",))
def nmf_multiplicative_update_H(
    H: jnp.ndarray, W: jnp.ndarray, V: jnp.ndarray, iterations: int, epsilon: float
) -> jnp.ndarray:
    """
    Perform multiple multiplicative updates on H matrix using JAX scan.

    This function updates H for a fixed number of iterations while keeping W constant.
    Uses the multiplicative update rule: H := H * (W.T @ V) / (W.T @ W @ H + epsilon)

    Args:
        H: Factor matrix to update, shape (n_components, n_features)
        W: Fixed factor matrix, shape (n_samples, n_components)
        V: Data matrix, shape (n_samples, n_features)
        iterations: Number of update steps to perform
        epsilon: Small value to prevent division by zero

    Returns:
        Updated H matrix of the same shape
    """

    def f(carry, x):
        H = carry

        H_new = jnp.clip(H * ((W.T @ V) / (W.T @ W @ H + epsilon)), epsilon, None)
        return (H_new, None)

    H_new, _ = jax.lax.scan(f, H, None, length=iterations)
    return H_new


@jax.jit
def update_H(
    W: jnp.ndarray, H: jnp.ndarray, V: jnp.ndarray, epsilon: float
) -> jnp.ndarray:
    """
    Perform a single multiplicative update step on H matrix.

    Args:
        W: Factor matrix, shape (n_samples, n_components)
        H: Factor matrix to update, shape (n_components, n_features)
        V: Data matrix, shape (n_samples, n_features)
        epsilon: Small value to prevent division by zero

    Returns:
        Updated H matrix of the same shape
    """
    return jnp.clip(H * ((W.T @ V) / (W.T @ W @ H + epsilon)), epsilon, None)


@partial(jax.jit, static_argnums=(3,))  # iterations should be static for scan to unroll
def _multiplicative_update_WH(
    W: jnp.ndarray,
    H: jnp.ndarray,
    V: jnp.ndarray,
    iterations: int,
    epsilon: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform multiple alternating multiplicative updates on both W and H matrices.

    This internal function performs the core NMF multiplicative update algorithm,
    alternating between updating H and W for the specified number of iterations.

    Args:
        W: Factor matrix, shape (n_samples, n_components)
        H: Factor matrix, shape (n_components, n_features)
        V: Data matrix, shape (n_samples, n_features)
        iterations: Number of update cycles to perform
        epsilon: Small value to prevent division by zero

    Returns:
        Tuple of (updated_W, updated_H)
    """

    def f(carry, x):
        # carry is (W, H) from the previous iteration
        W, H = carry
        H = jnp.clip(H * ((W.T @ V) / (W.T @ W @ H + epsilon)), epsilon, None)
        W = jnp.clip(W * ((V @ H.T) / (W @ (H @ H.T) + epsilon)), epsilon, None)
        return ((W, H), None)

    (W, H), _ = jax.lax.scan(f, (W, H), None, length=iterations)
    return (W, H)


def multiplicative_updates_H(
    V: jnp.ndarray,
    W: jnp.ndarray,
    H: jnp.ndarray,
    iterations: int = 10_000,
    verbose_frequency: int = 0,
    epsilon: float = 1e-12,
) -> Tuple[jnp.ndarray, List[Tuple[int, float, float, float]]]:
    """
    Perform NMF multiplicative updates on H matrix only (W is fixed).

    This function optimizes H while keeping W constant, useful for the
    "Minimize C(H|W,V)" step in alternating optimization schemes.

    Args:
        V: Data matrix to factorize, shape (n_samples, n_features)
        W: Fixed factor matrix, shape (n_samples, n_components)
        H: Initial factor matrix to optimize, shape (n_components, n_features)
        iterations: Total number of update iterations
        verbose_frequency: How often to print progress (0 = no progress output)
        epsilon: Small value to prevent division by zero

    Returns:
        Tuple of:
        - Updated H matrix of shape (n_components, n_features)
        - List of loss tracking tuples, each containing:
          (iteration, loss_value, max_abs_diff, mean_abs_diff)
    """

    if verbose_frequency == 0:
        epochs, n_steps_per_epoch = (1, iterations)
    else:
        epochs, n_steps_per_epoch = (iterations // verbose_frequency, verbose_frequency)

    desc = "Minimising C(H|W,V)"
    losses = (_, last_loss, *_) = [
        0,
        loss(W, H, V),
        max_abs_diff(W, H, V),
        mean_abs_diff(W, H, V),
    ]
    with tqdm(total=iterations, desc=f"{desc}: loss={last_loss}") as pb:
        for epoch in range(1, 1 + epochs):
            H = nmf_multiplicative_update_H(H, W, V, n_steps_per_epoch, epsilon=epsilon)

            this_loss, this_max_abs_diff, this_mean_abs_diff = (
                loss(W, H, V),
                max_abs_diff(W, H, V),
                mean_abs_diff(W, H, V),
            )
            losses.append(
                (
                    n_steps_per_epoch * epoch,
                    this_loss,
                    this_max_abs_diff,
                    this_mean_abs_diff,
                )
            )
            pb.set_description(
                f"{desc}: loss={this_loss:.2e} ({this_loss - last_loss:+.2e}); max_abs_diff={this_max_abs_diff:.2f}; mean_abs_diff={this_mean_abs_diff:.2e}"
            )
            last_loss = this_loss
            pb.update(n_steps_per_epoch)

    return (H, losses)


def multiplicative_updates_WH(
    V: jnp.ndarray,
    W: jnp.ndarray,
    H: jnp.ndarray,
    iterations: int = 10_000,
    verbose_frequency: int = 0,
    epsilon: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray, List[Tuple[int, float, float, float]]]:
    """
    Perform multiplicative updates for NMF on both W and H matrices.

    This function implements the standard NMF multiplicative update algorithm,
    alternating between updating H and W to minimize ||WH - V||_F^2.

    Args:
        V: Data matrix to factorize, shape (n_samples, n_features)
        W: Initial factor matrix, shape (n_samples, n_components)
        H: Initial factor matrix, shape (n_components, n_features)
        iterations: Total number of update iterations to perform
        verbose_frequency: How often to print progress (0 = no progress output)
        epsilon: Small value to prevent division by zero and ensure non-negativity

    Returns:
        Tuple of:
        - Updated W matrix of shape (n_samples, n_components)
        - Updated H matrix of shape (n_components, n_features)
        - List of loss tracking tuples, each containing:
          (iteration, loss_value, max_abs_diff, mean_abs_diff)

    Note:
        The multiplicative update rules used are:
        - H := H * (W.T @ V) / (W.T @ W @ H + epsilon)
        - W := W * (V @ H.T) / (W @ H @ H.T + epsilon)

        Both matrices are clipped to ensure non-negativity with minimum value epsilon.
    """

    if verbose_frequency == 0:
        epochs, n_steps_per_epoch = (1, iterations)
    else:
        epochs, n_steps_per_epoch = (iterations // verbose_frequency, verbose_frequency)

    desc = "Minimising C(W,H|V)"
    losses = (_, last_loss, *_) = [
        0,
        loss(W, H, V),
        max_abs_diff(W, H, V),
        mean_abs_diff(W, H, V),
    ]
    with tqdm(total=iterations, desc=f"{desc}: loss={last_loss}") as pb:
        for epoch in range(1, 1 + epochs):
            W, H = _multiplicative_update_WH(
                W, H, V, n_steps_per_epoch, epsilon=epsilon
            )

            this_loss, this_max_abs_diff, this_mean_abs_diff = (
                loss(W, H, V),
                max_abs_diff(W, H, V),
                mean_abs_diff(W, H, V),
            )
            losses.append(
                (
                    n_steps_per_epoch * epoch,
                    this_loss,
                    this_max_abs_diff,
                    this_mean_abs_diff,
                )
            )
            pb.set_description(
                f"{desc}: loss={this_loss:.2e} ({this_loss - last_loss:+.2e}); max_abs_diff={this_max_abs_diff:.2f}; mean_abs_diff={this_mean_abs_diff:.2e}"
            )
            last_loss = this_loss
            pb.update(n_steps_per_epoch)

    return (W, H, losses)
