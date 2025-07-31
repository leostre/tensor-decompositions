""""
Adapted from
https://github.com/OsmanMalik/tucker-tensorsketch/blob/master/tucker_ts.m
"""

from functools import reduce
from typing import Union, Tuple


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
    rmin = min(r)
    assert J1 <= rmin and J2 <= rmin
    # with torch.set_grad_enabled(): TODO

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
        As1_hat.append(fft(count_sketch(Q, h1[n], J1, s[n]), axis=1)) # CHANGED: Q not transposed
        print('Q', Q.size(), 'As1_hat', As1_hat[-1].size())
    
    # Compute sketches of input tensor
    YsT = []
    nnzY = len(torch.nonzero(Y))
    
    if verbose:
        print('Starting to compute sketches of input tensor...')
    
    if sflag:
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

    # elif extflag:
    #     # External computation case
    #     sketch_func = Y[0]
    #     if len(Y) > 2:
    #         sketch_params = (J1, J2, h1, h2, s, verbose, *Y[2:])
    #     else:
    #         sketch_params = (J1, J2, h1, h2, s, verbose)
        
    #     YsT, vecYs = sketch_func(*sketch_params)
    else:
        # Dense tensor case
        for n in range(n_dim):
            mode_n_unfolding = unfold(Y, n)
            YsT.append(tensor_sketch_mat(mode_n_unfolding, 
                                       [h1[i] for i in range(n_dim) if i != n],
                                       [s[i] for i in range(n_dim) if i != n],
                                       J1).T)
            if verbose:
                print(f'Finished computing sketch {n + 1} out of {n_dim + 1}...')
        
        vecYs = tensor_sketch_vec(Y, h2, s, J2)

    if verbose:
        print('Finished computing all sketches')
    
    # Main loop
    if verbose:
        print('Starting main loop...')
    
    norm_core = torch.norm(core)
    for iter in range(maxiters):
        norm_core_old = norm_core
        for n in range(n_dim):
            # TensorSketch the Kronecker product and compute sketched LS problem
            print('As1:', [
                t.size() for t in As1_hat
            ])
            kr_prod = khatri_rao([As1_hat[i].T for i in range(n_dim) if i != n])
            core_unfolding = unfold(core, n).to(torch.complex64)
            
            # Solve least squares problem
            mat = ifft((core_unfolding @ kr_prod).T, axis=1).float() 
            print(mat.size(), YsT[n].size())
            factor = torch.linalg.lstsq(mat, YsT[n]).solution.T
            
            # Orthogonalize factor matrix and update core tensor
            Q, R = torch.qr(factor)
            factors[n] = Q
            core = fold(R @ unfold(core, n), n, core.shape)
            
            # Update As1_hat[n]
            As1_hat[n] = fft(count_sketch(factor.T, h1[n], J1, s[n]), axis=1)
            factors[n] = factor
        
        # TensorSketch the Kronecker product using hash functions
        As2_hat = []
        for n in range(n_dim):
            As2_hat.append(fft(count_sketch(factors[n].T, h2[n], J2, s[n]), axis=1))
        
        M2 = ifft(khatri_rao(As2_hat).T, axis=1)
        M2tM2 = M2.T @ M2
        M2tvecYs = M2.T @ vecYs
        
        def matvec(x):
            return M2tM2 @ x

        core_vec, _ = torch.linalg.cg(
            A = matvec,     
            b = M2tvecYs,
            tol = 1e-5
        )
        core = torch.tensor(core_vec.reshape(core.shape), device=core.device)
        
        # Compute fit
        norm_core = torch.norm(core)
        norm_change = abs(norm_core - norm_core_old)
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

def tensor_sketch_mat(A, h_list, s_list, J):
    """Tensor sketch of matrix unfolding.
    
    Parameters
    ----------
    A : torch.Tensor
        Matrix unfolding of tensor (I x prod(I_other))
    h_list : list
        List of hash functions for each mode
    s_list : list
        List of sign functions for each mode
    J : int
        Sketch dimension
        
    Returns
    -------
    sketch : torch.Tensor
        Tensor sketch of A (J x prod(I_other))
    """
    # Count sketch along each mode
    sketches = []
    for h, s in zip(h_list, s_list):
        sketches.append(count_sketch(A, h, J, s)) # CHANGED: A not transposed
    # Multiply sketches element-wise
    
    return reduce(torch.mul, sketches)

def tensor_sketch_vec(Y, h_list, s_list, J):
    """Tensor sketch of tensor as vector.
    
    Parameters
    ----------
    Y : torch.Tensor
        Input tensor
    h_list : list
        List of hash functions for each mode
    s_list : list
        List of sign functions for each mode
    J : int
        Sketch dimension
        
    Returns
    -------
    sketch : torch.Tensor
        Tensor sketch of vec(Y) (J,)
    """
    # Flatten tensor
    vecY = Y.reshape(-1)
    
    # Compute combined hash and sign functions
    h_combined = 0
    s_combined = torch.ones(vecY.shape[0], device=vecY.device)
    
    stride = 1
    for n in range(len(h_list)):
        dim = Y.shape[n]
        indices = torch.arange(vecY.shape[0], device=vecY.device) // stride % dim
        h_combined = (h_combined + h_list[n][indices]) % J
        s_combined *= s_list[n][indices]
        stride *= dim
    
    # Apply count sketch
    return count_sketch(vecY.unsqueeze(1), h_combined, J, s_combined).squeeze()

# def sparse_tensor_sketch_mat(values, indices, h_list, s_list, J, In, n):
#     """Tensor sketch for sparse tensor along mode n."""
#     # This would need a custom C++/CUDA implementation for efficiency
#     # Placeholder implementation - would be very slow in Python
#     warnings.warn("Using slow Python implementation for sparse tensor sketch")
    
#     # Get the dimensions
#     other_dims = [h.shape[0] for h in h_list]
#     other_size = np.prod(other_dims)
    
#     # Initialize result
#     result = torch.zeros(J, In, device=values.device)
    
#     # Iterate through non-zero elements
#     for val, idx in zip(values, indices.T):
#         # Compute combined hash and sign
#         h = 0
#         s = 1.0
#         stride = 1
#         pos = 0
        
#         for i in range(len(indices)):
#             if i == n:
#                 in_idx = idx[i]
#                 continue
            
#             dim = h_list[pos].shape[0]
#             h = (h + h_list[pos][idx[i]]) % J
#             s *= s_list[pos][idx[i]]
#             pos += 1
        
#         # Update sketch
#         result[h, in_idx] += val * s
    
#     return result


# def sparse_tensor_sketch_vec(
#     values: torch.Tensor,
#     indices: torch.Tensor,
#     h: List[torch.Tensor],
#     s: List[torch.Tensor],
#     sketch_dim: int
# ) -> torch.Tensor:
#     """
#     Computes the TensorSketch of a sparse tensor's vectorization.
    
#     Args:
#         values: Non-zero values of the sparse tensor (1D tensor)
#         indices: Multi-dimensional indices of non-zero entries (2D tensor, shape [dims, nnz])
#         h: List of hash functions (int tensors), one per dimension
#         s: List of sign functions (float tensors), one per dimension
#         sketch_dim: Target sketch dimension
        
#     Returns:
#         Sketch vector of shape [sketch_dim]
#     """
#     # Validate inputs
#     assert len(values) == indices.shape[1], "Values and indices must match"
#     assert len(h) == len(s) == indices.shape[0], "Need one hash/sign function per dimension"
    
#     # Initialize output sketch vector
#     output_vec = torch.zeros(sketch_dim, dtype=values.dtype, device=values.device)
    
#     # Precompute hash and sign contributions
#     hash_contributions = []
#     sign_contributions = []
    
#     for dim in range(indices.shape[0]):
#         # Get indices for current dimension
#         dim_indices = indices[dim, :] - 1  # Convert to 0-based indexing
        
#         # Gather hash and sign values
#         hash_vals = h[dim][dim_indices.long()]
#         sign_vals = s[dim][dim_indices.long()]
        
#         hash_contributions.append(hash_vals)
#         sign_contributions.append(sign_vals)
    
#     # Compute combined hash and sign
#     combined_hash = torch.stack(hash_contributions).sum(dim=0)
#     combined_sign = torch.stack(sign_contributions).prod(dim=0)
    
#     # Compute target rows in sketch
#     target_rows = (combined_hash - len(h)) % sketch_dim
    
#     # Scatter-add the values
#     output_vec.index_add_(0, target_rows, values * combined_sign)
    
#     return output_vec


# def sparse_to_sparse_tensor_sketch_mat(values, indices, h_list, s_list, J, In, n):
#     """Alternative sparse tensor sketch that returns sparse result."""
#     # Placeholder implementation
#     warnings.warn("Using slow Python implementation for sparse tensor sketch")
    
#     # This would return (indices, values) for sparse matrix
#     # Actual implementation would need to be optimized
#     pass

# def sparse_tensor_sketch_vec(values, indices, h_list, s_list, J):
#     """Tensor sketch of sparse tensor as vector."""
#     # Placeholder implementation
#     warnings.warn("Using slow Python implementation for sparse tensor sketch")
    
#     # Compute combined hash and sign functions
#     h_combined = torch.zeros(indices.shape[1], dtype=torch.int64, device=values.device)
#     s_combined = torch.ones(indices.shape[1], device=values.device)
    
#     for i in range(len(h_list)):
#         h_combined = (h_combined + h_list[i][indices[i]]) % J
#         s_combined *= s_list[i][indices[i]]
    
#     # Apply count sketch
#     result = torch.zeros(J, device=values.device)
#     for h, s, val in zip(h_combined, s_combined, values):
#         result[h] += val * s
    
#     return result


if __name__ == '__main__':
    T = torch.randn(100, 100, 100, 100)
    r = (50, 1, 16, 100)
    core, factors = tucker_ts(T, r, 50, 50, verbose=True)
