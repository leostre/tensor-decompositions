""""
Adapted from
https://github.com/OsmanMalik/tucker-tensorsketch/blob/master/tucker_ts.m
"""

from functools import reduce
from typing import *


import torch
import tensorly as tl
from tensorly import decomposition
from tensorly.base import unfold, fold
from tensorly.tenalg import khatri_rao
# from scipy.sparse.linalg import LinearOperator, cg
import numpy as np
from torch.fft import fft, ifft
import warnings
# from time import time

from tdecomp.utils import svd_solver_tikhonov


tl.set_backend('pytorch')


def tucker_ts(Y: torch.Tensor, r, J1, J2, tol=1e-3, maxiters=50, verbose=False, backprop_compatible=False):
    """Implementation of one-pass TUCKER-TS algorithm using TensorSketch.
    
    Parameters
    ----------
    Y : torch.Tensor or tuple
        Input tensor or tuple (function_handle, size) for external sketch computation
    R : list or tuple
        Target Tucker rank
    J1 : int
        First sketch dimension
    J2 : int
        Second sketch dimension
    tol : float, optional
        Tolerance for convergence (default: 1e-3)
    maxiters : int, optional
        Maximum number of iterations (default: 50)
    verbose : bool, optional
        Whether to print progress information (default: False)
        
    Returns
    -------
    core : torch.Tensor
        Core tensor of the Tucker decomposition
    factors : list
        List of factor matrices
    """

    sizeY = Y.size()
    n_dim = Y.ndim 
    sflag = Y.is_sparse
    # Initialize hash functions
    h1 = []
    h2 = []
    s = []
    for n in range(n_dim):
        size = sizeY[n]
        h1.append(torch.randint(0, J1, (size,), dtype=torch.int32))
        h2.append(torch.randint(0, J2, (size,), dtype=torch.int32))
        s.append((torch.rand(size) > 0.5).int().mul_(2).sub_(1))
    
    # Initialize factor matrices and core tensor
    factors = []
    As1_hat = []
    
    # Initialize core and factor tensors with random values
    core = torch.rand(*r).mul_(2).sub_(1)

    for n in range(n_dim):
        Q = torch.linalg.qr(
                torch.rand(sizeY[n], r[n]).mul_(2).sub_(1),
                mode='reduced')[0] # size[n] x r[n]
        factors.append(Q)        
        # Compute count sketch
        As1_hat.append(fft(count_sketch(Q, h1[n], J1, s[n]), axis=1)) # CHANGED: Q transposed
    
    # Compute sketches of input tensor
    YsT = []
    nnzY = len(torch.nonzero(Y))
    
    if verbose:
        print('Starting to compute sketches of input tensor...')
    
    if sflag:
        # TODO from root repository
        def sparse_sketch(Y, YsT, nnzY):
            # Sparse tensor case
            for n in range(n_dim):
                if J1 * sizeY[n] < 3 * nnzY:
                    YsT.append(sparse_tensor_sketch_mat(
                        Y.values(), Y.indices(), 
                        [h1[i] for i in range(n_dim) if i != n], 
                        [s[i] for i in range(n_dim) if i != n],
                        J1, sizeY[n], n
                    ))
                else:
                    # For large sketch dimensions
                    subs, vals = sparse_to_sparse_tensor_sketch_mat(
                        Y.values(), Y.indices(),
                        [h1[i] for i in range(n_dim) if i != n],
                        [s[i] for i in range(n_dim) if i != n],
                        J1, sizeY[n], n
                    )
                    YsT.append(torch.sparse_coo_tensor(subs, vals, (J1, sizeY[n])))
            
                if verbose:
                    print(f'Finished computing sketch {n+1} out of {n_dim+1}...')
            
            vecYs = sparse_tensor_sketch_vec(
                Y.values(), Y.indices(), h2, s, J2
            )
            return YsT, vecYs
        YsT, vecYs = sparse_sketch(Y, YsT, nnzY)
    else:
        # Dense tensor case
        for n in range(n_dim):
            mode_n_unfolding = unfold(Y, n)
            # from TensorSketchMatC3_git
            YsT.append(tensor_sketch_mat(mode_n_unfolding, 
                                       [h1[i] for i in range(n_dim) if i != n],
                                       [s[i] for i in range(n_dim) if i != n],
                                       J1).T)
            if verbose:
                print(f'Finished computing sketch {n + 1} out of {n_dim + 1}...')
        
        vecYs = tensor_sketch_vec(Y, h2, s, J2)
        assert vecYs.shape == (J2,)

    if verbose:
        print('Finished computing all sketches')
    
    # Main loop
    if verbose:
        print('Starting main loop...')
    
    norm_core = torch.norm(core)
    for iter in range(maxiters):
        norm_core_old = norm_core
        for n in range(n_dim):
            kr_prod = khatri_rao([As1_hat[i].T for i in range(n_dim) if i != n])
            core_unfolding = unfold(core, n).to(torch.complex64)

            # Solve least squares problem
            mat = ifft((core_unfolding @ kr_prod).T, axis=1).real.T
            factor = torch.linalg.lstsq(mat.T, YsT[n]).solution        
            
            # Orthogonalize factor matrix and update core tensor
            Q, R = torch.linalg.qr(factor.T, mode='reduced')
            core = fold(R.T @ unfold(core, n), n, core.shape) # TODO check if R or R.T
            # Update As1_hat[n]
            As1_hat[n] = fft(count_sketch(Q, h1[n], J1, s[n]), axis=1)
            # print('As1', As1_hat[n].size())
            factors[n] = Q
        
        # TensorSketch the Kronecker product using hash functions
        As2_hat = []
        for n in range(n_dim):
            As2_hat.append(fft(count_sketch(factors[n], h2[n], J2, s[n]), axis=1).T)
        M2 = ifft(khatri_rao(As2_hat).T, axis=1).real

        core_vec = svd_solver_tikhonov(M2, vecYs)

        core = core_vec.reshape(core.shape)
        assert not torch.isnan(core).any()
        
        # Compute fit
        norm_core = torch.norm(core)
        norm_change = abs(norm_core - norm_core_old) / (norm_core_old + 1e-8)
        if verbose:
            print(f' Iter {iter+1:2d}: normChange = {norm_change:7.1e}')
        
        # Check for convergence
        if iter and norm_change < tol:
            break
    
    return core, factors

def count_sketch(A, h, J, s):
    """Count sketch of matrix A.
    
    Parameters
    ----------
    A : torch.Tensor
        Input matrix (d x m)
    h : torch.Tensor
        Hash functions (d,)
    s : torch.Tensor
        Sign functions (d,)
    J : int
        Sketch dimension
        
    Returns
    -------
    sketch : torch.Tensor
        Sketch of A (J x m)
    """
    device = A.device
    d, m = A.shape
    
    # Create sparse matrix for sketching
    indices = torch.stack([h, torch.randint(0, d, (len(h),), device=device)])
    values = s
    # print(indices.size(), values.size(), values.max(), indices.max(dim=-1))
    S = torch.sparse_coo_tensor(indices, s, (J, d), device=device, dtype=torch.float32)
    sketch = S @ A
    assert sketch.shape == (J, m)
    # print(sketch.size(), 'sketch')
    return sketch

# def tensor_sketch_mat(A, h_list, s_list, J):
#     """Tensor sketch of matrix unfolding.
    
#     Parameters
#     ----------
#     A : torch.Tensor
#         Matrix unfolding of tensor (I x prod(I_other))
#     h_list : list
#         List of hash functions for each mode
#     s_list : list
#         List of sign functions for each mode
#     J : int
#         Sketch dimension
        
#     Returns
#     -------
#     sketch : torch.Tensor
#         Tensor sketch of A (J x prod(I_other))
#     """
#     # Count sketch along each mode
#     sketches = []
#     for h, s in zip(h_list, s_list):
#         sketches.append(count_sketch(A, h, J, s)) # CHANGED: A not transposed
#     # Multiply sketches element-wise

def tensor_sketch_mat(
    M: torch.Tensor,
    h: List[torch.Tensor],
    s: List[torch.Tensor],
    sketch_dim: int,
    outer_dim_start: Optional[int] = None,
    outer_dim_end: Optional[int] = None
) -> torch.Tensor:
    """
    Further optimized version using scatter_add for better performance.
    """
    no_rows, no_cols = M.shape
    device = M.device
    
    # Handle partial sketching
    if outer_dim_start is not None and outer_dim_end is not None:
        partial_flag = True
        outer_dim_start = outer_dim_start - 1
        outer_dim_end = outer_dim_end - 1
    else:
        partial_flag = False
    
    # Generate all index combinations
    dim_sizes = [len(h_dim) for h_dim in h]
    if partial_flag:
        dim_sizes[-1] = outer_dim_end - outer_dim_start + 1
    
    # Create linear indices
    linear_indices = torch.arange(np.prod(dim_sizes), device=device)
    multi_indices = [torch.div(linear_indices, np.prod(dim_sizes[i+1:]), rounding_mode='trunc').int() % dim_sizes[i] 
                    for i in range(len(dim_sizes))]
    
    if partial_flag:
        multi_indices[-1] += outer_dim_start
    
    # Compute hash and sign
    combined_hash = sum(h_dim[idx] for h_dim, idx in zip(h, multi_indices))
    combined_sign = torch.prod(torch.stack([s_dim[idx] for s_dim, idx in zip(s, multi_indices)]), dim=0)
    
    # Compute target rows and column indices
    target_rows = (combined_hash - len(h)) % sketch_dim
    col_indices = torch.arange(no_cols, device=device).view(dim_sizes).flatten()
    
    # Scatter-add using index_add
    output_matrix = torch.zeros((no_rows, sketch_dim), dtype=M.dtype, device=device)
    for row in range(no_rows):
        output_matrix[row].index_add_(0, target_rows, M[row, col_indices] * combined_sign)
    
    return output_matrix

def tensor_sketch_vec(
    vec: torch.Tensor,
    h: List[torch.Tensor],
    s: List[torch.Tensor],
    sketch_dim: int,
    outer_dim_start: Optional[int] = None,
    outer_dim_end: Optional[int] = None
) -> torch.Tensor:
    """
    Vectorized TensorSketch implementation for PyTorch.
    
    Args:
        vec: Input vector of shape (prod(dim_sizes),)
        h: List of hash function tensors, each of shape (dim_size,)
        s: List of sign function tensors, each of shape (dim_size,)
        sketch_dim: Target sketch dimension size
        outer_dim_start: Optional starting index for partial sketching (1-based)
        outer_dim_end: Optional ending index for partial sketching (1-based)
        
    Returns:
        Sketch vector of shape (sketch_dim,)
    """
    device = vec.device
    dim_sizes = [h_dim.size(0) for h_dim in h]
    # total_elements = reduce(int.__mul__, dim_sizes)
    
    # Handle partial sketching
    if outer_dim_start is not None and outer_dim_end is not None:
        partial_flag = True
        outer_dim_start = outer_dim_start - 1  # Convert to 0-based
        outer_dim_end = outer_dim_end - 1
        dim_sizes[-1] = outer_dim_end - outer_dim_start + 1
    else:
        partial_flag = False
    
    # Create multi-dimensional indices
    indices = torch.meshgrid(*[torch.arange(size, device=device) for size in dim_sizes], indexing='ij')
    
    # Apply partial sketching offset if needed
    if partial_flag:
        indices[-1] = indices[-1] + outer_dim_start
    
    # Compute combined hash and sign
    combined_hash = sum(h_dim[idx] for h_dim, idx in zip(h, indices))
    combined_sign = torch.prod(torch.stack([s_dim[idx] for s_dim, idx in zip(s, indices)]), dim=0)
    
    # Compute target indices in sketch
    target_indices = (combined_hash - len(h)) % sketch_dim
    
    # Reshape input vector for partial sketching
    if partial_flag:
        # Calculate original flat indices
        strides = [1]
        for size in reversed(dim_sizes[1:]):
            strides.insert(0, strides[0] * size)
        strides = torch.tensor(strides, device=device)
        
        # Compute original indices
        original_indices = sum(idx * stride for idx, stride in zip(indices, strides))
        vec_values = vec[original_indices.flatten()]
    else:
        vec_values = vec.reshape(-1)
    
    # Initialize sketch vector
    sketch = torch.zeros(sketch_dim, dtype=vec.dtype, device=device)
    
    # Vectorized scatter-add
    sketch.index_add_(0, target_indices.flatten(), vec_values * combined_sign.flatten())
    
    return sketch

if __name__ == '__main__':
    T = torch.randn(100, 100, 100, 100)
    r = (50, 1, 16, 100)
    core, factors = tucker_ts(T, r, 50, 50, verbose=True)
