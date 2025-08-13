import jax
import jax.numpy as jnp
from functools import reduce
from typing import List, Tuple
import numpy as np

def create_1d_fourier_modes(n_samples: int, n_modes: int) -> jnp.ndarray:
    """
    Create the mode indices for 1D real Fourier basis.
    
    Returns array of shape (n_modes, 2) where each row is [frequency, type]
    type: 0 = constant, 1 = cosine, 2 = sine
    """
    modes = []
    
    # Constant term
    if n_modes > 0:
        modes.append([0, 0])  # freq=0, type=constant
    
    # Add cosine/sine pairs
    freq = 1
    while len(modes) < n_modes:
        if len(modes) < n_modes:
            modes.append([freq, 1])  # cosine
        if len(modes) < n_modes:
            modes.append([freq, 2])  # sine
        freq += 1
    
    return jnp.array(modes[:n_modes])

def evaluate_1d_fourier_basis(x: jnp.ndarray, modes: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate 1D Fourier basis functions at points x.
    
    Args:
        x: sampling points, shape (n_samples,)
        modes: mode specification, shape (n_modes, 2)
    
    Returns:
        basis matrix of shape (n_samples, n_modes)
    """
    n_samples = x.shape[0]
    n_modes = modes.shape[0]
    
    # Vectorized evaluation
    freqs = modes[:, 0]  # shape (n_modes,)
    types = modes[:, 1]  # shape (n_modes,)
    
    # Broadcast: x is (n_samples, 1), freqs is (1, n_modes)
    x_expanded = x[:, None]  # (n_samples, 1)
    freqs_expanded = freqs[None, :]  # (1, n_modes)
    
    # Compute all frequency-point combinations
    phase = freqs_expanded * x_expanded  # (n_samples, n_modes)
    
    # Apply the appropriate function based on type
    basis = jnp.where(types == 0, 1.0,  # constant
                     jnp.where(types == 1, jnp.cos(phase),  # cosine
                               jnp.sin(phase)))  # sine
    
    return basis

def matvec(samples_per_dim: List[int], 
                     modes_per_dim: List[int], 
                     x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute A @ x where A is an N-dimensional Fourier design matrix.
    
    Uses the separable structure: the N-D transform is a sequence of 1-D transforms.
    
    Args:
        samples_per_dim: number of samples in each dimension
        modes_per_dim: number of Fourier modes in each dimension  
        x: coefficient vector, shape (prod(modes_per_dim),)
    
    Returns:
        result vector, shape (prod(samples_per_dim),)
    """
    n_dims = len(samples_per_dim)
    
    # Reshape x to tensor form: (modes_0, modes_1, ..., modes_{n_dims-1})
    x_tensor = x.reshape(modes_per_dim)
    
    # Create sampling grids for each dimension
    coords = []
    for i, n_samples in enumerate(samples_per_dim):
        coord = jnp.linspace(0, 2*jnp.pi, n_samples, endpoint=False)
        coords.append(coord)
    
    # Apply separable transform: transform along each dimension sequentially
    result = x_tensor
    
    for dim in range(n_dims):
        # Get modes for this dimension
        modes = create_1d_fourier_modes(samples_per_dim[dim], modes_per_dim[dim])
        
        # Create 1D basis matrix for this dimension
        basis_1d = evaluate_1d_fourier_basis(coords[dim], modes)  # (n_samples, n_modes)
        
        # Apply transformation along this dimension
        # We need to contract along the current dimension
        # Move the dimension to be transformed to the last axis
        result = jnp.moveaxis(result, dim, -1)
        
        # Reshape for matrix multiplication: (..., modes_dim) -> (..., samples_dim)
        original_shape = result.shape
        result_2d = result.reshape(-1, original_shape[-1])  # (batch, modes_dim)
        
        # Apply 1D transform: (batch, modes) @ (modes, samples)^T = (batch, samples)
        result_2d = result_2d @ basis_1d.T
        
        # Reshape back and move dimension back to original position
        new_shape = original_shape[:-1] + (samples_per_dim[dim],)
        result = result_2d.reshape(new_shape)
        result = jnp.moveaxis(result, -1, dim)
    
    # Flatten to vector
    return result.flatten()

def rmatvec(samples_per_dim: List[int], 
                      modes_per_dim: List[int], 
                      y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute A.T @ y where A is an N-dimensional Fourier design matrix.
    
    Args:
        samples_per_dim: number of samples in each dimension
        modes_per_dim: number of Fourier modes in each dimension
        y: input vector, shape (prod(samples_per_dim),)
    
    Returns:
        result vector, shape (prod(modes_per_dim),)
    """
    n_dims = len(samples_per_dim)
    
    # Reshape y to tensor form: (samples_0, samples_1, ..., samples_{n_dims-1})
    y_tensor = y.reshape(samples_per_dim)
    
    # Create sampling grids for each dimension
    coords = []
    for i, n_samples in enumerate(samples_per_dim):
        coord = jnp.linspace(0, 2*jnp.pi, n_samples, endpoint=False)
        coords.append(coord)
    
    # Apply adjoint separable transform
    result = y_tensor
    
    for dim in range(n_dims):
        # Get modes for this dimension
        modes = create_1d_fourier_modes(samples_per_dim[dim], modes_per_dim[dim])
        
        # Create 1D basis matrix for this dimension
        basis_1d = evaluate_1d_fourier_basis(coords[dim], modes)  # (n_samples, n_modes)
        
        # Apply adjoint transformation along this dimension
        # Move the dimension to be transformed to the last axis
        result = jnp.moveaxis(result, dim, -1)
        
        # Reshape for matrix multiplication: (..., samples_dim) -> (..., modes_dim)
        original_shape = result.shape
        result_2d = result.reshape(-1, original_shape[-1])  # (batch, samples_dim)
        
        # Apply 1D adjoint transform: (batch, samples) @ (samples, modes) = (batch, modes)
        result_2d = result_2d @ basis_1d
        
        # Reshape back and move dimension back to original position
        new_shape = original_shape[:-1] + (modes_per_dim[dim],)
        result = result_2d.reshape(new_shape)
        result = jnp.moveaxis(result, -1, dim)
    
    # Flatten to vector
    return result.flatten()

def eval_at_point(point: jnp.ndarray,
                           modes_per_dim: List[int],
                           coefficients: jnp.ndarray) -> float:
    """
    Evaluate the Fourier series at a specific N-dimensional point.
    
    This computes: sum_i c_i * φ_i(point) where φ_i are the N-D Fourier basis functions.
    Equivalent to computing (A @ coefficients)[point_index] if point was in the sampling grid.
    
    Args:
        point: N-dimensional coordinates, shape (n_dims,)
        modes_per_dim: number of Fourier modes in each dimension
        coefficients: Fourier coefficients, shape (prod(modes_per_dim),)
    
    Returns:
        scalar value of the Fourier series at the given point
    """
    n_dims = len(modes_per_dim)
    
    # Reshape coefficients to tensor form
    coeff_tensor = coefficients.reshape(modes_per_dim)
    
    # Compute 1D basis evaluations at the point coordinates
    basis_values_1d = []
    
    for dim in range(n_dims):
        coord = point[dim]
        n_modes = modes_per_dim[dim]
        
        # Get modes for this dimension
        modes = create_1d_fourier_modes(n_modes, n_modes)  # Note: using n_modes for both args
        
        # Evaluate 1D basis functions at this coordinate
        basis_1d = evaluate_1d_fourier_basis(jnp.array([coord]), modes)  # (1, n_modes)
        basis_values_1d.append(basis_1d[0, :])  # Extract the single row: (n_modes,)
    
    # The N-D basis evaluation is the tensor product of 1D evaluations
    # We need to contract the coefficient tensor with the basis evaluations
    result = coeff_tensor
    
    for dim in range(n_dims):
        # Contract along dimension 'dim' with the basis values for that dimension
        result = jnp.tensordot(result, basis_values_1d[dim], axes=([0], [0]))
        # After contraction, the remaining dimensions shift down
    
    return result

def eval_basis_at_point(point: jnp.ndarray,
                                 modes_per_dim: List[int]) -> jnp.ndarray:
    """
    Evaluate all N-dimensional Fourier basis functions at a specific point.
    
    This gives you one row of the design matrix A corresponding to the given point.
    
    Args:
        point: N-dimensional coordinates, shape (n_dims,)
        modes_per_dim: number of Fourier modes in each dimension
    
    Returns:
        basis values at the point, shape (prod(modes_per_dim),)
    """
    n_dims = len(modes_per_dim)
    
    # Compute 1D basis evaluations at each coordinate
    basis_values_1d = []
    
    for dim in range(n_dims):
        coord = point[dim]
        n_modes = modes_per_dim[dim]
        
        # Get modes for this dimension
        modes = create_1d_fourier_modes(n_modes, n_modes)
        
        # Evaluate 1D basis functions at this coordinate
        basis_1d = evaluate_1d_fourier_basis(jnp.array([coord]), modes)  # (1, n_modes)
        basis_values_1d.append(basis_1d[0, :])  # (n_modes,)
    
    # Compute tensor product of all 1D basis evaluations
    # This gives us all combinations of basis functions
    result = basis_values_1d[0]  # Start with first dimension: (modes_0,)
    
    for dim in range(1, n_dims):
        # Take outer product with next dimension
        result = jnp.outer(result, basis_values_1d[dim])  # (..., modes_dim)
        result = result.flatten()  # Flatten to 1D
    
    return result

def gram_diagonal(
    samples_per_dim: List[int], 
    modes_per_dim: List[int],
    mask: jnp.array = None
) -> jnp.ndarray:
    """
    Compute the diagonal of A.T @ A where A is an N-dimensional Fourier design matrix.
    
    Uses the fact that for separable bases, the Gram matrix diagonal is the 
    Kronecker product of 1D Gram matrix diagonals.
    
    Args:
        samples_per_dim: number of samples in each dimension
        modes_per_dim: number of Fourier modes in each dimension
        mask: optional mask to apply to the diagonal computation, shape (prod(samples_per_dim), )
    
    Returns:
        diagonal vector, shape (prod(modes_per_dim),)
    """
    if mask is None:
        mask = jnp.ones(reduce(lambda x, y: x * y, samples_per_dim), dtype=bool)

    mask = mask.astype(jnp.int32)

    n_dims = len(samples_per_dim)
    
    # Compute 1D Gram matrix diagonals for each dimension
    gram_diagonals_1d = []
    
    for dim in range(n_dims):
        n_samples = samples_per_dim[dim]
        n_modes = modes_per_dim[dim]
        
        # Create coordinate array for this dimension
        coord = jnp.linspace(0, 2*jnp.pi, n_samples, endpoint=False)
        
        # Get modes for this dimension
        modes = create_1d_fourier_modes(n_samples, n_modes)
        
        # Compute 1D basis matrix
        basis_1d = evaluate_1d_fourier_basis(coord, modes)  # (n_samples, n_modes)
        
        # Compute diagonal of 1D Gram matrix
        axis = tuple(i for i in range(len(samples_per_dim)) if i != dim)
        # TODO: should this be 1/weights?
        weights = 1/jnp.mean(mask, axis=axis).reshape((-1, 1))
        gram_diag_1d = jnp.sum(basis_1d * weights * basis_1d, axis=0)  # (n_modes,)
        gram_diagonals_1d.append(gram_diag_1d)
    
    # The N-D Gram matrix diagonal is the Kronecker product of 1D diagonals
    # For diagonals, Kronecker product becomes outer products
    gram_diagonal_nd = gram_diagonals_1d[0]
    
    for dim in range(1, n_dims):
        # Compute outer product with next dimension
        gram_diagonal_nd = jnp.outer(gram_diagonal_nd, gram_diagonals_1d[dim])
        gram_diagonal_nd = gram_diagonal_nd.flatten()
    
    return gram_diagonal_nd

def matdiag(samples_per_dim: List[int], 
                      modes_per_dim: List[int], 
                      diagonal_elements: jnp.ndarray) -> jnp.ndarray:
    """
    Compute A @ M where A is an N-dimensional Fourier design matrix and M is a diagonal matrix.
    
    This is equivalent to element-wise multiplication of each column of A with the corresponding
    diagonal element. Uses the separable structure to avoid constructing the full matrix A.
    
    Mathematically: (A @ M)_{i,j} = A_{i,j} * M_{j,j} = A_{i,j} * diagonal_elements[j]
    
    Args:
        samples_per_dim: number of samples in each dimension
        modes_per_dim: number of Fourier modes in each dimension  
        diagonal_elements: diagonal elements of M, shape (prod(modes_per_dim),)
    
    Returns:
        result matrix A @ M, shape (prod(samples_per_dim), prod(modes_per_dim))
        Returned as flattened columns: [col_0, col_1, ..., col_{n_modes-1}]
    """
    n_dims = len(samples_per_dim)
    n_total_modes = np.prod(modes_per_dim)
    n_total_samples = np.prod(samples_per_dim)
    
    # Create sampling grids for each dimension
    coords = []
    for i, n_samples in enumerate(samples_per_dim):
        coord = jnp.linspace(0, 2*jnp.pi, n_samples, endpoint=False)
        coords.append(coord)
    
    # We'll compute A @ M column by column, where each column j corresponds to 
    # the j-th basis function scaled by diagonal_elements[j]
    result_columns = []
    
    for mode_idx in range(n_total_modes):
        # Convert linear mode index to N-D mode indices
        mode_indices = jnp.unravel_index(mode_idx, modes_per_dim)
        
        # Create a coefficient vector that is 1 at this mode and 0 elsewhere
        coeff = jnp.zeros(n_total_modes)
        coeff = coeff.at[mode_idx].set(1.0)
        
        # Compute A @ coeff to get the mode_idx-th column of A
        column = matvec(samples_per_dim, modes_per_dim, coeff)
        
        # Scale by the diagonal element
        scaled_column = column * diagonal_elements[mode_idx]
        
        result_columns.append(scaled_column)
    
    # Stack columns to form the result matrix
    # Shape: (n_total_samples, n_total_modes)
    result_matrix = jnp.column_stack(result_columns)
    
    return result_matrix

def matdiag_efficient(samples_per_dim: List[int], 
                                modes_per_dim: List[int], 
                                diagonal_elements: jnp.ndarray) -> jnp.ndarray:
    """
    More efficient version of matdiag that uses the separable structure directly.
    
    Instead of computing each column separately, we use the fact that the N-D basis functions
    are tensor products of 1-D basis functions, so we can compute the scaled result more efficiently.
    """
    n_dims = len(samples_per_dim)
    
    # Reshape diagonal elements to tensor form
    diag_tensor = diagonal_elements.reshape(modes_per_dim)
    
    # Create sampling grids and 1D basis matrices for each dimension
    coords = []
    basis_matrices_1d = []
    
    for dim in range(n_dims):
        n_samples = samples_per_dim[dim]
        n_modes = modes_per_dim[dim]
        
        # Create coordinate array for this dimension
        coord = jnp.linspace(0, 2*jnp.pi, n_samples, endpoint=False)
        coords.append(coord)
        
        # Get modes and compute 1D basis matrix
        modes = create_1d_fourier_modes(n_samples, n_modes)
        basis_1d = evaluate_1d_fourier_basis(coord, modes)  # (n_samples, n_modes)
        basis_matrices_1d.append(basis_1d)
    
    # The key insight: A @ M where M is diagonal can be computed as:
    # We apply the diagonal scaling to the coefficient tensor, then do the forward transform
    
    # Apply separable transform with diagonal scaling
    result = diag_tensor
    
    for dim in range(n_dims):
        # Apply transformation along this dimension
        result = jnp.moveaxis(result, dim, -1)
        
        # Reshape for matrix multiplication
        original_shape = result.shape
        result_2d = result.reshape(-1, original_shape[-1])  # (batch, modes_dim)
        
        # Apply 1D transform: (batch, modes) @ (modes, samples)^T = (batch, samples)
        result_2d = result_2d @ basis_matrices_1d[dim].T
        
        # Reshape back and move dimension back to original position
        new_shape = original_shape[:-1] + (samples_per_dim[dim],)
        result = result_2d.reshape(new_shape)
        result = jnp.moveaxis(result, -1, dim)
    
    # Result now has shape (samples_0, samples_1, ..., samples_{n_dims-1})
    # We need to return it as a matrix of shape (prod(samples_per_dim), prod(modes_per_dim))
    
    # The efficient approach: we've computed the result of applying A @ diag(diagonal_elements)
    # But we need to return the matrix A @ M, not A @ M applied to a vector
    
    # Actually, let's rethink this. The user wants A @ M as a matrix operation.
    # The efficient way is to use the column-wise approach but vectorize it properly.
    
    # Use vmap to compute all columns in parallel
    def compute_scaled_column(mode_idx):
        # Create unit vector for this mode
        coeff = jnp.zeros(np.prod(modes_per_dim))
        coeff = coeff.at[mode_idx].set(diagonal_elements[mode_idx])
        
        # Apply A to get scaled column
        return matvec(samples_per_dim, modes_per_dim, coeff)
    
    # Vectorize over all mode indices
    mode_indices = jnp.arange(np.prod(modes_per_dim))
    result_columns = jax.vmap(compute_scaled_column)(mode_indices)
    
    # Transpose to get (n_samples, n_modes) matrix
    return result_columns.T

def matdiag_efficient(samples_per_dim: List[int], 
                                modes_per_dim: List[int], 
                                diagonal_elements: jnp.ndarray) -> jnp.ndarray:
    """
    More efficient version of matdiag that uses the separable structure directly.
    
    Instead of computing each column separately, we use the fact that the N-D basis functions
    are tensor products of 1-D basis functions, so we can compute the scaled result more efficiently.
    """
    n_dims = len(samples_per_dim)
    
    # Reshape diagonal elements to tensor form
    diag_tensor = diagonal_elements.reshape(modes_per_dim)
    
    # Create sampling grids and 1D basis matrices for each dimension
    coords = []
    basis_matrices_1d = []
    
    for dim in range(n_dims):
        n_samples = samples_per_dim[dim]
        n_modes = modes_per_dim[dim]
        
        # Create coordinate array for this dimension
        coord = jnp.linspace(0, 2*jnp.pi, n_samples, endpoint=False)
        coords.append(coord)
        
        # Get modes and compute 1D basis matrix
        modes = create_1d_fourier_modes(n_samples, n_modes)
        basis_1d = evaluate_1d_fourier_basis(coord, modes)  # (n_samples, n_modes)
        basis_matrices_1d.append(basis_1d)
    
    # The key insight: A @ M where M is diagonal can be computed as:
    # We apply the diagonal scaling to the coefficient tensor, then do the forward transform
    
    # Apply separable transform with diagonal scaling
    result = diag_tensor
    
    for dim in range(n_dims):
        # Apply transformation along this dimension
        result = jnp.moveaxis(result, dim, -1)
        
        # Reshape for matrix multiplication
        original_shape = result.shape
        result_2d = result.reshape(-1, original_shape[-1])  # (batch, modes_dim)
        
        # Apply 1D transform: (batch, modes) @ (modes, samples)^T = (batch, samples)
        result_2d = result_2d @ basis_matrices_1d[dim].T
        
        # Reshape back and move dimension back to original position
        new_shape = original_shape[:-1] + (samples_per_dim[dim],)
        result = result_2d.reshape(new_shape)
        result = jnp.moveaxis(result, -1, dim)
    
    # Result now has shape (samples_0, samples_1, ..., samples_{n_dims-1})
    # We need to return it as a matrix of shape (prod(samples_per_dim), prod(modes_per_dim))
    
    # The efficient approach: we've computed the result of applying A @ diag(diagonal_elements)
    # But we need to return the matrix A @ M, not A @ M applied to a vector
    
    # Actually, let's rethink this. The user wants A @ M as a matrix operation.
    # The efficient way is to use the column-wise approach but vectorize it properly.
    
    # Use vmap to compute all columns in parallel
    def compute_scaled_column(mode_idx):
        # Create unit vector for this mode
        coeff = jnp.zeros(np.prod(modes_per_dim))
        coeff = coeff.at[mode_idx].set(diagonal_elements[mode_idx])
        
        # Apply A to get scaled column
        return matvec(samples_per_dim, modes_per_dim, coeff)
    
    # Vectorize over all mode indices
    mode_indices = jnp.arange(np.prod(modes_per_dim))
    result_columns = jax.vmap(compute_scaled_column)(mode_indices)
    
    # Transpose to get (n_samples, n_modes) matrix
    return result_columns.T

def hvag_inverse(samples_per_dim: List[int], 
                           modes_per_dim: List[int], 
                           H: jnp.ndarray,
                           V: jnp.ndarray) -> jnp.ndarray:
    """
    Compute H @ V.T @ A @ (A.T @ A)^(-1) efficiently without materializing large matrices.
    
    This is useful for computing preconditioned projections where:
    - A is the N-dimensional Fourier design matrix
    - (A.T @ A) is diagonal (Gram matrix) 
    - V and H are auxiliary matrices
    
    The computation is done as: H @ (V.T @ (A @ (A.T @ A)^(-1)))
    
    Args:
        samples_per_dim: number of samples in each dimension
        modes_per_dim: number of Fourier modes in each dimension  
        H: left multiplication matrix, shape (n_H, prod(samples_per_dim))
        V: right multiplication matrix, shape (prod(samples_per_dim), n_V)
    
    Returns:
        result matrix, shape (n_H, prod(modes_per_dim))
    """
    # Step 1: Compute diagonal of A.T @ A
    gram_diagonal = gram_diagonal_jit(samples_per_dim, modes_per_dim)
    
    # Step 2: Compute (A.T @ A)^(-1) - since it's diagonal, just take reciprocal
    gram_inv_diagonal = 1.0 / gram_diagonal
    
    # Step 3: Compute A @ (A.T @ A)^(-1) using our matdiag function
    # This gives us A @ diag((A.T @ A)^(-1))
    A_gram_inv = matdiag_efficient_jit(samples_per_dim, modes_per_dim, gram_inv_diagonal)
    # Shape: (prod(samples_per_dim), prod(modes_per_dim))
    
    # Step 4: Compute V.T @ (A @ (A.T @ A)^(-1))
    VT_A_gram_inv = V.T @ A_gram_inv
    # Shape: (n_V, prod(modes_per_dim))
    
    # Step 5: Compute H @ (V.T @ A @ (A.T @ A)^(-1))
    result = H @ VT_A_gram_inv
    # Shape: (n_H, prod(modes_per_dim))
    
    return result

def hvag_inverse_efficient(samples_per_dim: List[int], 
                                     modes_per_dim: List[int], 
                                     H: jnp.ndarray,
                                     V: jnp.ndarray,
                                     M: jnp.ndarray) -> jnp.ndarray:
    """
    More efficient version that avoids materializing A @ (A.T @ A)^(-1).
    
    Instead of computing the full matrix A @ (A.T @ A)^(-1), we use the fact that
    this operation can be done column-wise using matrix-vector products.
    
    The key insight: (H @ V.T @ A @ (A.T @ A)^(-1))[:, j] = 
                     H @ V.T @ A @ ((A.T @ A)^(-1) * e_j)
                   = H @ V.T @ A @ (e_j / gram_diagonal[j])
                   = (H @ V.T) @ (A @ (e_j / gram_diagonal[j]))
    """
    # Step 1: Compute diagonal of A.T @ A
    gram_diagonal = gram_diagonal_jit(samples_per_dim, modes_per_dim, M)
    
    # Step 2: Precompute H @ V.T 
    HVT = H @ V.T  # Shape: (n_H, prod(samples_per_dim))
    
    # Step 3: For each column j, compute HVT @ (A @ (e_j / gram_diagonal[j]))
    n_modes = np.prod(modes_per_dim)
    
    def compute_column(mode_idx):
        # Create scaled unit vector: e_j / gram_diagonal[j]
        scaled_unit = jnp.zeros(n_modes)
        scaled_unit = scaled_unit.at[mode_idx].set(1.0 / gram_diagonal[mode_idx])
        
        # Compute A @ scaled_unit
        # TODO: Do we need to apply the mask M here?
        # If M is provided, we can apply it to the scaled unit
        A_scaled_unit = matvec_jit(samples_per_dim, modes_per_dim, scaled_unit)
        
        # Compute HVT @ (A @ scaled_unit)
        return HVT @ A_scaled_unit
    
    # Vectorize over all mode indices
    mode_indices = jnp.arange(n_modes)
    result_columns = jax.vmap(compute_column)(mode_indices)
    
    # Transpose to get (n_H, n_modes) matrix
    return result_columns.T

def hvag_inverse_memory_efficient(samples_per_dim: List[int], 
                                            modes_per_dim: List[int], 
                                            H: jnp.ndarray,
                                            V: jnp.ndarray,
                                            chunk_size: int = 1000) -> jnp.ndarray:
    """
    Memory-efficient version that processes columns in chunks to handle very large problems.
    
    This is useful when the number of modes is very large and computing all columns at once
    would exceed memory limits.
    """
    # Step 1: Compute diagonal of A.T @ A
    gram_diagonal = gram_diagonal_jit(samples_per_dim, modes_per_dim)
    
    # Step 2: Precompute H @ V.T 
    HVT = H @ V.T  # Shape: (n_H, prod(samples_per_dim))
    
    # Step 3: Process columns in chunks
    n_modes = np.prod(modes_per_dim)
    n_H = H.shape[0]
    
    # Initialize result matrix
    result = jnp.zeros((n_H, n_modes))
    
    # Process in chunks
    for start_idx in range(0, n_modes, chunk_size):
        end_idx = min(start_idx + chunk_size, n_modes)
        chunk_indices = jnp.arange(start_idx, end_idx)
        
        def compute_chunk_column(mode_idx):
            # Create scaled unit vector: e_j / gram_diagonal[j]
            scaled_unit = jnp.zeros(n_modes)
            scaled_unit = scaled_unit.at[mode_idx].set(1.0 / gram_diagonal[mode_idx])
            
            # Compute A @ scaled_unit
            A_scaled_unit = matvec_jit(samples_per_dim, modes_per_dim, scaled_unit)
            
            # Compute HVT @ (A @ scaled_unit)
            return HVT @ A_scaled_unit
        
        # Compute chunk of columns
        chunk_columns = jax.vmap(compute_chunk_column)(chunk_indices)
        
        # Update result
        result = result.at[:, start_idx:end_idx].set(chunk_columns.T)
    
    return result

# JIT compile the functions for performance
matvec_jit = jax.jit(matvec, static_argnums=(0, 1))
rmatvec_jit = jax.jit(rmatvec, static_argnums=(0, 1))
gram_diagonal_jit = jax.jit(gram_diagonal, static_argnums=(0, 1))
eval_at_point_jit = jax.jit(eval_at_point, static_argnums=(1,))
eval_basis_at_point_jit = jax.jit(eval_basis_at_point, static_argnums=(1,))
matdiag_jit = jax.jit(matdiag, static_argnums=(0, 1))
matdiag_efficient_jit = jax.jit(matdiag_efficient, static_argnums=(0, 1))
hvag_inverse_jit = jax.jit(hvag_inverse, static_argnums=(0, 1))
hvag_inverse_efficient_jit = jax.jit(hvag_inverse_efficient, static_argnums=(0, 1))
# Note: memory_efficient version not JITted due to chunk_size parameter

matmat_jit = jax.vmap(matvec_jit, in_axes=(None, None, 1))
rmatmat_jit = jax.vmap(rmatvec_jit, in_axes=(None, None, 1))