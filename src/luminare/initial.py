import jax
import jax.numpy as jnp
from functools import partial
from jaxopt import ProjectedGradient, BoxCDQP
from jaxopt.projection import projection_box

import numpy as np
from scipy.optimize._lsq.trf_linear import trf_linear


def create_initial_estimator(H, A, large=jnp.inf):

    A_full = jnp.hstack([H, A])

    def initial(flux, ivar):

        bounds = (
            -large * jnp.ones(A_full.shape[1]),
            jnp.hstack([jnp.zeros(H.shape[1]), +large * jnp.ones(A.shape[1])])
        )
        use = jnp.any(A != 0, axis=1) * (ivar > 0)
        Aw = (A_full * jnp.sqrt(ivar[:, None]))
        Yw = flux * jnp.sqrt(ivar)
        
        # TODO: Would be good to have a jax version of this.
        f = partial(np.array, dtype=np.float64)
        x0 = jnp.linalg.lstsq(Aw, Yw, rcond=None)[0]
        x0 = jnp.clip(x0, *bounds)
        x0, Aw, Yw, (lb, ub) = map(f, (x0, Aw, Yw, bounds))
        eps = np.finfo(float).eps
        kwds = dict(tol=eps, lsmr_tol=eps, lsq_solver="exact", max_iter=10_000, verbose=0)

        r = trf_linear(Aw[use], Yw[use], x_lsq=x0, lb=lb, ub=ub, **kwds)

        θ_c = r.x[-A.shape[1]:]
        continuum = A @ θ_c

        Hw = H * (jnp.sqrt(ivar) * continuum)[:, None]
        Yw = (continuum - flux) * jnp.sqrt(ivar)

        # TODO: Would be good to have a jax version of this.
        lb, ub = (jnp.zeros(H.shape[1]), jnp.inf * jnp.ones(H.shape[1]))
        x0 = jnp.clip(jnp.linalg.lstsq(Hw[use], Yw[use], rcond=None)[0], lb, ub)
        x0, Hw, Yw = map(f, (x0, Hw, Yw))
        r = trf_linear(Hw, Yw, x_lsq=x0, lb=lb, ub=ub, **kwds)
        θ_W = r.x

        rectified_flux = 1 - H @ θ_W
        return (θ_W, θ_c, rectified_flux, continuum)
        
    return initial


#@partial(jax.jit, static_argnames=("index", "n", "max_vsini"))
def estimate_vsini(model, flux, ivar, θ_initial, index, max_vsini=400, n=20):

    @jax.jit
    def trial_vsini(vsini):
        θ = jnp.copy(θ_initial)
        θ = θ.at[index].set(vsini)
        return jnp.sum((model(θ) - flux)**2 * ivar)

    vsinis = jnp.linspace(0, max_vsini, n)
    chi2 = jax.vmap(trial_vsini)(vsinis)
    vsini_best = vsinis[jnp.argmin(chi2)]
    return (vsini_best, vsinis, chi2)