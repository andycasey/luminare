import jax
import jax.numpy as jnp
from jax import jit
from typing import NamedTuple, Optional, Tuple, Union
import functools

# Constants
EPS = jnp.finfo(jnp.float64).eps

class OptimizeResult(NamedTuple):
    """Result object for optimization."""
    x: jnp.ndarray
    fun: jnp.ndarray
    cost: float
    optimality: float
    active_mask: jnp.ndarray
    nit: int
    status: int
    initial_cost: float

# Utility functions (JAX implementations of scipy common functions)

#@jit
def step_size_to_bound(x, s, lb, ub):
    """Compute step size to bound."""
    non_zero = jnp.abs(s) > EPS
    
    # Steps to lower bounds
    steps_to_lb = jnp.where(
        non_zero & (s < 0),
        (lb - x) / s,
        jnp.inf
    )
    
    # Steps to upper bounds  
    steps_to_ub = jnp.where(
        non_zero & (s > 0),
        (ub - x) / s,
        jnp.inf
    )
    
    steps = jnp.minimum(steps_to_lb, steps_to_ub)
    min_step = jnp.min(steps)
    hits = (steps <= min_step + EPS) & non_zero
    
    return min_step, hits

#@jit
def find_active_constraints(x, lb, ub, rtol=1e-10):
    """Find which constraints are active."""
    active_l = jnp.abs(x - lb) < rtol * jnp.maximum(1, jnp.abs(lb))
    active_u = jnp.abs(x - ub) < rtol * jnp.maximum(1, jnp.abs(ub))
    
    return jnp.where(active_l, -1, jnp.where(active_u, 1, 0))

#@jit
def in_bounds(x, lb, ub):
    """Check if point is within bounds."""
    return jnp.all((x >= lb) & (x <= ub))

#@jit
def make_strictly_feasible(x, lb, ub, rstep=1e-10):
    """Move point strictly inside bounds."""
    x_proj = jnp.clip(x, lb, ub)
    tight_bounds = (ub - lb) <= rstep
    
    # For tight bounds, set to midpoint
    x_proj = jnp.where(tight_bounds, 0.5 * (lb + ub), x_proj)
    
    # Move away from bounds
    active_lower = (x_proj <= lb + rstep) & ~tight_bounds
    active_upper = (x_proj >= ub - rstep) & ~tight_bounds
    
    x_proj = jnp.where(active_lower, lb + rstep, x_proj)
    x_proj = jnp.where(active_upper, ub - rstep, x_proj)
    
    return x_proj

#@jit
def reflective_transformation(y, lb, ub):
    """Apply reflective transformation to keep point in bounds."""
    def reflect_single(yi, lbi, ubi):
        if lbi == -jnp.inf and ubi == jnp.inf:
            return yi, 1.0
        elif lbi == -jnp.inf:
            return jnp.where(yi <= ubi, yi, 2*ubi - yi), jnp.where(yi <= ubi, 1.0, -1.0)
        elif ubi == jnp.inf:
            return jnp.where(yi >= lbi, yi, 2*lbi - yi), jnp.where(yi >= lbi, 1.0, -1.0)
        else:
            width = ubi - lbi
            yi_shifted = yi - lbi
            
            # Number of complete reflections
            n_complete = jnp.floor(yi_shifted / width)
            remainder = yi_shifted - n_complete * width
            
            # Direction after n_complete reflections
            direction = jnp.where(n_complete % 2 == 0, 1.0, -1.0)
            
            x_transformed = jnp.where(
                direction > 0,
                lbi + remainder,
                ubi - remainder
            )
            
            return x_transformed, direction
    
    x_new = jnp.zeros_like(y)
    g_sign = jnp.zeros_like(y)
    
    for i in range(len(y)):
        x_new = x_new.at[i].set(reflect_single(y[i], lb[i], ub[i])[0])
        g_sign = g_sign.at[i].set(reflect_single(y[i], lb[i], ub[i])[1])
    
    return x_new, g_sign

#@jit
def build_quadratic_1d(J_h, g_h, s, diag=None, s0=None):
    """Build 1D quadratic function coefficients."""
    if s0 is not None:
        s = s0 + s
    
    Js = J_h @ s
    if diag is not None:
        s_diag = s * diag
        a = 0.5 * (jnp.dot(Js, Js) + jnp.dot(s, s_diag))
        b = jnp.dot(g_h, s) + jnp.dot(s_diag, s0) if s0 is not None else jnp.dot(g_h, s)
    else:
        a = 0.5 * jnp.dot(Js, Js)
        b = jnp.dot(g_h, s)
    
    return a, b, 0.0

#@jit
def minimize_quadratic_1d(a, b, lb, ub, c=0):
    """Minimize 1D quadratic function."""
    if a == 0:
        return jnp.where(b >= 0, lb, ub), -jnp.inf
    
    minimum = -b / (2 * a)
    t = jnp.clip(minimum, lb, ub)
    value = a * t**2 + b * t + c
    
    return t, value

#@jit
def evaluate_quadratic(J, g, s, diag=None):
    """Evaluate quadratic function."""
    Js = J @ s
    quad = 0.5 * jnp.dot(Js, Js) + jnp.dot(g, s)
    
    if diag is not None:
        quad += 0.5 * jnp.dot(s * diag, s)
    
    return quad

#@jit
def compute_grad(J, r):
    """Compute gradient J^T @ r."""
    return J.T @ r

#@jit
def CL_scaling_vector(x, g, lb, ub):
    """Compute Cauchy-Lovász scaling vector."""
    v = jnp.ones_like(x)
    dv = jnp.zeros_like(x)
    
    # Finite bounds case
    finite_lb = jnp.isfinite(lb)
    finite_ub = jnp.isfinite(ub)
    both_finite = finite_lb & finite_ub
    
    # Both bounds finite
    d = jnp.where(both_finite, ub - lb, 1.0)
    t = jnp.where(both_finite, (x - lb) / d, 0.5)
    
    # Scale based on gradient direction and proximity to bounds
    t_min = 0.01
    t_max = 0.99
    
    v = jnp.where(
        both_finite,
        jnp.minimum(t, 1-t) / jnp.maximum(t_min, jnp.minimum(t_max, jnp.minimum(t, 1-t))),
        1.0
    )
    
    # Only lower bound finite
    only_lb = finite_lb & ~finite_ub
    t_lb = jnp.where(only_lb & (g > 0), jnp.maximum(1.0, g), 1.0)
    v = jnp.where(only_lb, t_lb, v)
    
    # Only upper bound finite  
    only_ub = ~finite_lb & finite_ub
    t_ub = jnp.where(only_ub & (g < 0), jnp.maximum(1.0, -g), 1.0)
    v = jnp.where(only_ub, t_ub, v)
    
    # Derivatives
    dv = jnp.where(
        both_finite,
        (1 - 2*t) / (d * jnp.maximum(t_min, jnp.minimum(t_max, jnp.minimum(t, 1-t)))**2),
        0.0
    )
    
    dv = jnp.where(only_lb & (g > 0), jnp.sign(g), dv)
    dv = jnp.where(only_ub & (g < 0), -jnp.sign(g), dv)
    
    return v, dv

# Givens elimination implementation
#@jit
def givens_elimination(R, v, diag):
    """Apply Givens rotations to eliminate diagonal regularization."""
    m, n = R.shape
    
    # Work on copies
    R_work = R.copy()
    v_work = v.copy()
    
    for i in range(n):
        if diag[i] != 0:
            # Apply Givens rotation
            a = R_work[i, i]
            b = diag[i]
            
            # Compute rotation
            if b == 0:
                c, s = 1.0, 0.0
            elif jnp.abs(b) > jnp.abs(a):
                t = a / b
                s = 1.0 / jnp.sqrt(1 + t*t)
                c = s * t
            else:
                t = b / a
                c = 1.0 / jnp.sqrt(1 + t*t)
                s = c * t
            
            # Apply rotation to R
            for j in range(i, n):
                temp = c * R_work[i, j] + s * diag[i] if j == i else c * R_work[i, j]
                R_work = R_work.at[i, j].set(temp)
            
            # Apply rotation to v
            temp = c * v_work[i]
            v_work = v_work.at[i].set(temp)
    
    return R_work, v_work

#@jit
def regularized_lsq_with_qr(m, n, R, QTb, perm, diag, copy_R=True):
    """Solve regularized least squares using QR decomposition."""
    R_work = jnp.where(copy_R, R.copy(), R)
    v = QTb.copy()
    
    # Apply Givens elimination
    R_work, v = givens_elimination(R_work, v, diag[perm])
    
    # Determine numerical rank
    abs_diag_R = jnp.abs(jnp.diag(R_work))
    threshold = EPS * max(m, n) * jnp.max(abs_diag_R)
    nns = abs_diag_R > threshold
    
    # Create indices for non-null space
    indices = jnp.arange(n)
    valid_indices = indices[nns]
    n_valid = jnp.sum(nns)
    
    # Extract submatrices - pad to avoid empty arrays
    max_size = jnp.maximum(n_valid, 1)
    
    # Safe indexing with padding
    valid_indices_padded = jnp.concatenate([valid_indices, jnp.zeros(max_size - n_valid, dtype=jnp.int32)])
    
    R_sub = R_work[jnp.ix_(valid_indices_padded[:max_size], valid_indices_padded[:max_size])]
    v_sub = v[valid_indices_padded[:max_size]]
    
    # Solve triangular system only for valid part
    def solve_system():
        x_sub = jax.scipy.linalg.solve_triangular(
            R_sub[:n_valid, :n_valid], 
            v_sub[:n_valid], 
            lower=False
        )
        return x_sub
    
    def return_zeros():
        return jnp.zeros(0)
    
    x_sub = jax.lax.cond(n_valid > 0, solve_system, return_zeros)
    
    # Reconstruct full solution
    x = jnp.zeros(n)
    
    # Only update if we have valid solutions
    def update_solution(x):
        perm_valid = perm[valid_indices[:n_valid]]
        return x.at[perm_valid].set(x_sub)
    
    def keep_zeros(x):
        return x
        
    x = jax.lax.cond(n_valid > 0, update_solution, keep_zeros, x)
    
    return x

#@jit
def backtracking(A, g, x, p, theta, p_dot_g, lb, ub):
    """Backtracking line search."""
    alpha = 1.0
    x_new = x
    step = jnp.zeros_like(x)
    cost_change = 0.0
    
    # Simple backtracking
    for _ in range(10):  # Max 10 iterations
        x_trial, _ = reflective_transformation(x + alpha * p, lb, ub)
        step_trial = x_trial - x
        cost_change_trial = -evaluate_quadratic(A, g, step_trial)
        
        if cost_change_trial >= 0.1 * alpha * (-p_dot_g):
            x_new = x_trial
            step = step_trial
            cost_change = cost_change_trial
            break
        alpha *= 0.5
    
    return x_new, step, cost_change

#@jit  
def select_step(x, A_h, g_h, c_h, p, p_h, d, lb, ub, theta):
    """Select best step using Trust Region Reflective algorithm."""
    if in_bounds(x + p, lb, ub):
        return p
    
    # Find step to boundary
    p_stride, hits = step_size_to_bound(x, p, lb, ub)
    
    # Reflected direction
    r_h = p_h.copy()
    r_h = jnp.where(hits, -r_h, r_h)
    r = d * r_h
    
    # Restrict step to boundary
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p
    
    # Step along reflected direction
    r_stride_u, _ = step_size_to_bound(x_on_bound, r, lb, ub)
    r_stride_l = (1 - theta) * r_stride_u
    r_stride_u *= theta
    
    if r_stride_u > 0:
        a, b, c = build_quadratic_1d(A_h, g_h, r_h, s0=p_h, diag=c_h)
        r_stride, r_value = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c=c)
        r_h = p_h + r_h * r_stride
        r = d * r_h
    else:
        r_value = jnp.inf
    
    # Corrected step
    p_h *= theta
    p *= theta
    p_value = evaluate_quadratic(A_h, g_h, p_h, diag=c_h)
    
    # Anti-gradient direction
    ag_h = -g_h
    ag = d * ag_h
    ag_stride_u, _ = step_size_to_bound(x, ag, lb, ub)
    ag_stride_u *= theta
    a, b = build_quadratic_1d(A_h, g_h, ag_h, diag=c_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride_u)
    ag *= ag_stride
    
    # Choose best step
    if p_value <= r_value and p_value <= ag_value:
        return p
    elif r_value <= p_value and r_value <= ag_value:
        return r
    else:
        return ag

#@functools.partial(jit, static_argnames=['max_iter', 'lsq_solver'])
@jax.disable_jit()
def trf_linear(A, b, x_lsq, lb, ub, tol=1e-10, lsq_solver='exact', 
               lsmr_tol=None, max_iter=None, verbose=0, lsmr_maxiter=None):
    """
    JAX implementation of Trust Region Reflective algorithm for linear least squares.
    
    Parameters
    ----------
    A : array_like, shape (m, n)
        Design matrix
    b : array_like, shape (m,)  
        Target vector
    x_lsq : array_like, shape (n,)
        Unconstrained least squares solution
    lb : array_like, shape (n,)
        Lower bounds
    ub : array_like, shape (n,)
        Upper bounds
    tol : float, optional
        Tolerance for termination
    lsq_solver : str, optional
        Method for solving least squares subproblems ('exact' only for JAX)
    lsmr_tol : float, optional
        Not used in JAX implementation
    max_iter : int, optional
        Maximum number of iterations
    verbose : int, optional
        Verbosity level (not used in JAX)
    lsmr_maxiter : int, optional
        Not used in JAX implementation
        
    Returns
    -------
    result : OptimizeResult
        Optimization result
    """
    m, n = A.shape
    
    # Initialize
    x, _ = reflective_transformation(x_lsq, lb, ub)
    x = make_strictly_feasible(x, lb, ub, rstep=0.1)
    
    # QR decomposition for exact solver
    Q, R = jnp.linalg.qr(A, mode='reduced')
    QT = Q.T
    perm = jnp.arange(n)  # No pivoting in JAX QR
    
    if m < n:
        R = jnp.vstack([R, jnp.zeros((n - m, n))])
    
    QTr = jnp.zeros(n)
    k = min(m, n)
    
    # Initial cost and gradient
    r = A @ x - b
    g = compute_grad(A, r)
    cost = 0.5 * jnp.dot(r, r)
    initial_cost = cost
    
    # Termination status
    termination_status = 0
    step_norm = 0.0
    cost_change = 0.0
    
    if max_iter is None:
        max_iter = 100
    
    # Main iteration loop
    def iteration_body(carry):
        x, r, g, cost, termination_status, step_norm, cost_change, iteration, QTr = carry
        
        # Scaling
        v, dv = CL_scaling_vector(x, g, lb, ub)
        g_scaled = g * v
        g_norm = jnp.linalg.norm(g_scaled, ord=jnp.inf)
        
        # Check termination
        termination_status = jnp.where(g_norm < tol, 1, termination_status)
        
        # Scaled problem
        diag_h = g * dv
        diag_root_h = jnp.sqrt(jnp.abs(diag_h))
        d = jnp.sqrt(v)
        g_h = d * g
        
        # Solve subproblem
        A_h = A * d[None, :]  # Right multiplication
        QTr = QTr.at[:k].set(QT @ r)
        p_h = -regularized_lsq_with_qr(m, n, R * d[perm], QTr, perm, diag_root_h, copy_R=False)
        p = d * p_h
        
        # Check descent direction
        p_dot_g = jnp.dot(p, g)
        termination_status = jnp.where(p_dot_g > 0, -1, termination_status)
        
        # Step selection
        theta = 1 - jnp.minimum(0.005, g_norm)
        step = select_step(x, A_h, g_h, diag_h, p, p_h, d, lb, ub, theta)
        cost_change = -evaluate_quadratic(A, g, step)
        
        # Backtracking if needed
        x_new, step_new, cost_change_new = jax.lax.cond(
            cost_change < 0,
            lambda: backtracking(A, g, x, p, theta, p_dot_g, lb, ub),
            lambda: (x, step, cost_change)
        )
        
        # Update
        x = make_strictly_feasible(x_new + step_new, lb, ub, rstep=0)
        step_norm = jnp.linalg.norm(step_new)
        r = A @ x - b
        g = compute_grad(A, r)
        
        # Check cost change termination - use relative change like scipy
        new_cost = 0.5 * jnp.dot(r, r)
        relative_change = jnp.abs(cost - new_cost) / jnp.max(jnp.array([cost, new_cost, 1e-10]))
        termination_status = jnp.where(relative_change < tol, 2, termination_status)
        cost = new_cost
        
        return x, r, g, cost, termination_status, step_norm, cost_change_new, iteration + 1, QTr
    
    def continuation_condition(carry):
        _, _, _, _, termination_status, _, _, iteration, _ = carry
        return (termination_status == 0) & (iteration < max_iter)
    
    # Run iterations
    initial_carry = (x, r, g, cost, termination_status, step_norm, cost_change, 0, QTr)
    final_carry = jax.lax.while_loop(continuation_condition, iteration_body, initial_carry)

    x, r, g, cost, termination_status, step_norm, cost_change, nit, QTr = final_carry

    # Final gradient norm for optimality
    v, _ = CL_scaling_vector(x, g, lb, ub)
    g_scaled = g * v
    g_norm = jnp.linalg.norm(g_scaled, ord=jnp.inf)
    
    # Active constraints
    active_mask = find_active_constraints(x, lb, ub, rtol=tol)
    
    return OptimizeResult(
        x=x,
        fun=r,
        cost=cost,
        optimality=g_norm,
        active_mask=active_mask,
        nit=nit,
        status=termination_status,
        initial_cost=initial_cost
    )

# Example usage and test
if __name__ == "__main__":
    # Test problem
    jnp.random.seed(42)
    m, n = 10, 5
    A = jax.random.normal(jax.random.PRNGKey(42), (m, n))
    x_true = jax.random.normal(jax.random.PRNGKey(43), (n,))
    b = A @ x_true + 0.01 * jax.random.normal(jax.random.PRNGKey(44), (m,))
    
    # Bounds
    lb = jnp.full(n, -2.0)
    ub = jnp.full(n, 2.0)
    
    # Unconstrained solution
    x_lsq = jnp.linalg.lstsq(A, b, rcond=None)[0]
    
    # Solve with TRF
    result = trf_linear(A, b, x_lsq, lb, ub)
    
    print(f"Solution: {result.x}")
    print(f"Cost: {result.cost}")
    print(f"Optimality: {result.optimality}")
    print(f"Iterations: {result.nit}")
    print(f"Status: {result.status}")