from functools import partial
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=['max_iter', 'tol', 'update_H'])
def fit_coordinate_descent(X, W_init, H_init, l1_reg_H=0.0, l2_reg_H=0.0, 
                          l1_reg_W=0.0, l2_reg_W=0.0, max_iter=200, tol=1e-4,
                          random_state=0, update_H=True):
    """
    JAX implementation of NMF coordinate descent fitting.
    
    Args:
        X: Data matrix (n_samples, n_features)
        W_init: Initial basis matrix (n_samples, n_components)
        H_init: Initial coefficient matrix (n_components, n_features)
        l1_reg_H, l2_reg_H: L1 and L2 regularization for H
        l1_reg_W, l2_reg_W: L1 and L2 regularization for W  
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        random_state: Random seed
        update_H: Whether to update H matrix (default: True)
        
    Returns:
        Tuple of (W, H, n_iter, converged)
    """
    
    # Initialize
    W = W_init
    Ht = H_init.T  # Work with transposed H for efficiency
    
    # Generate random permutations for shuffling coordinates
    key = jax.random.PRNGKey(random_state)
    n_samples, n_features = X.shape
    shuffle_indices = jax.random.permutation(key, n_samples)
    
    def compute_objective(W_curr, H_curr):
        """Compute the NMF objective function."""
        reconstruction_error = jnp.sum((X - W_curr @ H_curr) ** 2)
        l1_penalty = l1_reg_W * jnp.sum(jnp.abs(W_curr)) + l1_reg_H * jnp.sum(jnp.abs(H_curr))
        
        # Update each component
        def update_component(j, h_curr):
            # Gradient computation
            grad = -W_curr[:, j].T @ (residual + W_curr[:, j] * h_curr[j])
            
            # Add L2 regularization
            grad += l2_reg_H * h_curr[j]
            
            # Compute denominator for coordinate update
            denom = W_curr[:, j].T @ W_curr[:, j] + l2_reg_H
            
            # Coordinate descent update with L1 regularization (soft thresholding)
            numerator = h_curr[j] - grad / denom
            new_val = jnp.maximum(0.0, numerator - l1_reg_H / denom)
            new_val = jnp.minimum(new_val, numerator + l1_reg_H / denom)
            
            return h_curr.at[j].set(new_val)
        
        # Update all components for this feature
        h_updated = jax.lax.fori_loop(0, h_idx.shape[0], update_component, h_idx)
        Ht_updated = Ht_curr.at[idx, :].set(h_updated)
        
        return W_curr, Ht_updated
    
    def update_W_single_sample(i, carry):
        W_curr, Ht_curr = carry
        idx = shuffle_indices[i]
        
        # Current basis vector for sample idx
        w_idx = W_curr[idx, :]
        
        # Residual without current sample contribution
        residual = X[idx, :] - w_idx @ Ht_curr.T
        
        # Update each component
        def update_component(j, w_curr):
            # Gradient computation
            grad = -Ht_curr[:, j].T @ (residual + Ht_curr[:, j] * w_curr[j])
            
            # Add L2 regularization
            grad += l2_reg_W * w_curr[j]
            
            # Compute denominator for coordinate update
            denom = Ht_curr[:, j].T @ Ht_curr[:, j] + l2_reg_W
            
            # Coordinate descent update with L1 regularization (soft thresholding)
            numerator = w_curr[j] - grad / denom
            new_val = jnp.maximum(0.0, numerator - l1_reg_W / denom)
            new_val = jnp.minimum(new_val, numerator + l1_reg_W / denom)
            
            return w_curr.at[j].set(new_val)
        
        # Update all components for this sample
        w_updated = jax.lax.fori_loop(0, w_idx.shape[0], update_component, w_idx)
        W_updated = W_curr.at[idx, :].set(w_updated)
        
        return W_updated, Ht_curr
    
    # Conditionally update H or W
    if update_H:
        n_features = X.shape[1]
        W_new, Ht_new = jax.lax.fori_loop(0, n_features, update_H_single_feature, (W, Ht))
        return W_new, Ht_new
    else:
        n_samples = X.shape[0]
        W_new, Ht_new = jax.lax.fori_loop(0, n_samples, update_W_single_sample, (W, Ht))
        return W_new, Ht_new

