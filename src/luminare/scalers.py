import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple


class BaseScaler(eqx.Module):
    pass

class NoScaler(BaseScaler):

    ndim: int = 0

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return x
    
class PeriodicScaler(BaseScaler):
    """A scaler to transform values to a periodic domain."""

    n: jax.Array
    minimum: jax.Array
    maximum: jax.Array
    ndim: int

    def __init__(
        self,
        n: Optional[jax.Array] = None,
        minimum: Optional[jax.Array] = None,
        maximum: Optional[jax.Array] = None,
    ):
        self.n = n
        self.minimum = minimum
        self.maximum = maximum
        self.ndim = len(n) if n is not None else 0

    def fit(self, X: Tuple[jnp.ndarray, ...]) -> "PeriodicScaler":
        if not isinstance(X, tuple):
            X = (X, )
        m = lambda f: jnp.array(list(map(f, X)))
        return PeriodicScaler(n=m(len), minimum=m(jnp.min), maximum=m(jnp.max))

    def fit_transform(self, X):
        self = self.fit(X)
        return self(X)

    def transform(self, x):
        x = (x - self.minimum) / (self.maximum - self.minimum)
        domain_max = 2 * jnp.pi * (self.n - 1) / self.n
        return x * domain_max

    def inverse_transform(self, x):
        # The "domain_max + edge" is the wrap-around point.
        domain_max = 2 * jnp.pi * (self.n - 1) / self.n
        edge = (2 * jnp.pi - domain_max) / 2

        x %= 2 * jnp.pi
        x = jnp.where(x > (domain_max + edge), x - 2 * jnp.pi, x) / domain_max
        return x * (self.maximum - self.minimum) + self.minimum

    def __call__(self, X):
        return self.transform(X)
