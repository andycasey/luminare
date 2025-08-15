import jax
import jax.numpy as jnp
from typing import Tuple


def periodic_scalers(n: jax.Array, minimum: jax.Array, maximum: jax.Array):
    """
    Create a pair of functions to transform and inverse-transform data to a 
    perioidic domain.

    Args:
        n: Number of points in each dimension.
        minimum: Minimum value of the data.
        maximum: Maximum value of the data.
    Returns:
        A tuple of two functions: transform and inverse_transform.
    """
    
    def transform(x):
        x = (x - minimum) / (maximum - minimum)
        domain_max = 2 * jnp.pi * (n - 1) / n
        return x * domain_max

    def inverse_transform(x):
        domain_max = 2 * jnp.pi * (n - 1) / n
        edge = (2 * jnp.pi - domain_max) / 2

        x %= 2 * jnp.pi
        x = jnp.where(x > (domain_max + edge), x - 2 * jnp.pi, x) / domain_max
        return x * (maximum - minimum) + minimum

    return (transform, inverse_transform)
    
def fit_periodic_scalers(X: Tuple[jnp.ndarray, ...]):
    """
    Create a pair of functions to transform and inverse-transform periodic data.
    
    Args:
        X: Tuple of arrays representing the real-valued data.
    Returns:
        A tuple of two functions: transform and inverse_transform.
    """
    
    if not isinstance(X, tuple):
        X = (X, )
    mapper = lambda f: jnp.array(list(map(f, X)))
    n, minimum, maximum = map(mapper, (len, jnp.min, jnp.max))

    return periodic_scalers(n, minimum, maximum)
