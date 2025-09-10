import jax
import jax.numpy as jnp
from functools import partial
from jaxopt import ProjectedGradient, BoxCDQP
from jaxopt.projection import projection_box

import numpy as np
from scipy.optimize import lsq_linear

def solve_bounded_lstsq(
    A, 
    b, 
    bounds_lower, 
    bounds_upper, 
    x0=None, 
    maxiter=1000,
    stepsize=1e-4,
    maxls=50,
    tol=1e-12,
    **kwargs
):
    """
    Solve bounded linear least squares: min ||Ax - b||^2 subject to bounds
    Using BoxCDQP - closest analog to scipy's trf_linear algorithm
    """
        
    if x0 is None:
        x0 = jnp.zeros(A.shape[1])
    
    # Convert linear least squares to QP form: min 0.5 x^T Q x + c^T x
    Q = A.T @ A
    c = -A.T @ b
    objective = lambda x: jnp.sum((A @ x - b)**2)
    
    # BoxCDQP is the closest to trf_linear - coordinate descent for box-constrained QP
    kwds = dict(maxiter=maxiter, tol=tol)
    kwds.update(kwargs)
    solver = BoxCDQP(maxiter=1e3, tol=1e-2, verbose=1, implicit_diff=False)#jit=False, **kwds)
    
    result = solver.run(
        x0,
        params_obj=(Q, c),
        params_ineq=(bounds_lower, bounds_upper),
    )
    print(objective(x0), objective(result.params))
    raise a
    return result



def create_initial_estimator(H, A, large=jnp.inf):

    use = jnp.any(A != 0, axis=1)

    A_full = jnp.hstack([H, A])
    bounds = (
        -large * jnp.ones(A_full.shape[1]),
        jnp.hstack([jnp.zeros(H.shape[1]), +large * jnp.ones(A.shape[1])])
    )

    def initial(flux, ivar):
        ATCinv = (A_full[use] * ivar[use][:, None]).T
        ATCinvA = ATCinv @ A_full[use]
        ATCinvY = ATCinv @ flux[use]
        
        # TODO: Would be good to have a jax version of this.
        f = partial(np.array, dtype=np.float64)
        r = lsq_linear(*map(f, (ATCinvA, ATCinvY, bounds)))

        θ_c = r.x[-A.shape[1]:]
        continuum = A @ θ_c
        
        Y = 1 - flux / continuum
        Cinv = ivar * continuum**2
        bad = ~jnp.isfinite(Y) | (Cinv == 0)
        Y = jnp.where(bad, 0.0, Y)
        Cinv = jnp.where(bad, 0.0, Cinv)
        HTCinv = (H * Cinv[:, None]).T

        # TODO: Would be good to have a jax version of this.
        r = lsq_linear(
            *map(f, (HTCinv @ H, HTCinv @ Y)), 
            bounds=(0, jnp.inf)
        )
        θ_W = r.x
        rectified_flux = 1 - H @ θ_W
        return (θ_W, θ_c, rectified_flux, continuum)
        
    return initial