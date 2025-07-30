""""
Adapted from
https://github.com/OsmanMalik/tucker-tensorsketch/blob/master/tucker_ts.m
"""

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


def compute_sketches(factors, R, N, sizeY, As1_hat, h1, J1, s):
    for n in range(N):
        # Initialize factor matrices with random values
        factors.append(torch.rand(sizeY[n], R[n]) * 2 - 1)
        
        # Orthogonalize
        Q, _ = torch.qr(factors[n], )
        factors[n] = Q
        
        # Compute count sketch
        As1_hat.append(fft(count_sketch(factors[n].T, h1[n], J1, s[n]), axis=1))
    
    # Compute sketches of input tensor
    YsT = []
    
    if verbose:
        print('Starting to compute sketches of input tensor...')
    
    if sflag:
        # Sparse tensor case
        for n in range(N):
            if J1 * sizeY[n] < 3 * nnzY:
                YsT.append(sparse_tensor_sketch_mat(
                    Y.values(), Y.indices(), 
                    [h1[i] for i in range(N) if i != n], 
                    [s[i] for i in range(N) if i != n],
                    J1, sizeY[n], n
                ))
            else:
                # For large sketch dimensions
                subs, vals = sparse_to_sparse_tensor_sketch_mat(
                    Y.values(), Y.indices(),
                    [h1[i] for i in range(N) if i != n],
                    [s[i] for i in range(N) if i != n],
                    J1, sizeY[n], n
                )
                YsT.append(torch.sparse_coo_tensor(subs, vals, (J1, sizeY[n])))
            
            if verbose:
                print(f'Finished computing sketch {n+1} out of {N+1}...')
        
        vecYs = sparse_tensor_sketch_vec(
            Y.values(), Y.indices(), h2, s, J2
        )
    elif extflag:
        # External computation case
        sketch_func = Y[0]
        if len(Y) > 2:
            sketch_params = (J1, J2, h1, h2, s, verbose, *Y[2:])
        else:
            sketch_params = (J1, J2, h1, h2, s, verbose)
        
        YsT, vecYs = sketch_func(*sketch_params)
    else:
        # Dense tensor case
        for n in range(N):
            mode_n_unfolding = unfold(Y, n)
            YsT.append(tensor_sketch_mat(mode_n_unfolding, 
                                       [h1[i] for i in range(N) if i != n],
                                       [s[i] for i in range(N) if i != n],
                                       J1).T)
            if verbose:
                print(f'Finished computing sketch {n+1} out of {N+1}...')
        
        vecYs = tensor_sketch_vec(Y, h2, s, J2)
    


def tucker_ts(Y: torch.Tensor, R, J1, J2, tol=1e-3, maxiters=50, verbose=False, backprop_compatible=False):
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
    # with torch.set_grad_enabled(): TODO
    
    N = len(sizeY) 

    sflag = Y.layout is torch.empty((4,), layout='sparse')
    
    # Initialize hash functions
    h1 = []
    h2 = []
    s = []
    for n in range(N):
        h1.append(torch.randint(0, J1, (sizeY[n],), dtype=torch.int64))
        h2.append(torch.randint(0, J2, (sizeY[n],), dtype=torch.int64))
        s.append((torch.rand(sizeY[n]) > 0.5).float() * 2 - 1)
    
    # Initialize factor matrices and core tensor
    factors = []
    As1_hat = []
    
    # Initialize core tensor with random values
    core = torch.rand(*R) * 2 - 1

    compute
    if verbose:
        print('Finished computing all sketches')
    
    # Main loop
    if verbose:
        print('Starting main loop...')
    
    normG = torch.norm(core)
    for iter in range(maxiters):
        normG_old = normG
        
        for n in range(N):
            # TensorSketch the Kronecker product and compute sketched LS problem
            kr_prod = khatri_rao([As1_hat[i] for i in range(N) if i != n])
            core_unfolding = unfold(core, n)
            
            # Solve least squares problem
            mat = ifft((core_unfolding @ kr_prod.T).T, axis=1)
            factors[n] = torch.linalg.lstsq(mat, YsT[n]).solution.T
            
            # Orthogonalize factor matrix and update core tensor
            Q, R = torch.qr(factors[n])
            factors[n] = Q
            core = fold(R @ unfold(core, n), n, core.shape)
            
            # Update As1_hat[n]
            As1_hat[n] = fft(count_sketch(factors[n].T, h1[n], J1, s[n]), axis=1)
        
        # TensorSketch the Kronecker product using hash functions
        As2_hat = []
        for n in range(N):
            As2_hat.append(fft(count_sketch(factors[n].T, h2[n], J2, s[n]), axis=1))
        
        # Compute sketched LS problem using conjugate gradient
        M2 = ifft(khatri_rao(As2_hat).T, axis=1)
        M2tM2 = M2.T @ M2
        M2tvecYs = M2.T @ vecYs
        
        # # Use scipy's conjugate gradient since PyTorch's is less stable
        # def matvec(x):
        #     return M2tM2 @ x
        
        # A = LinearOperator(shape=(M2tM2.shape[0], M2tM2.shape[1]), matvec=matvec)
        # core_vec, _ = cg(A, M2tvecYs.cpu().numpy())
        core = torch.tensor(core_vec.reshape(core.shape), device=core.device)
        
        # Compute fit
        normG = torch.norm(core)
        normChange = abs(normG - normG_old)
        if verbose:
            print(f' Iter {iter+1:2d}: normChange = {normChange:7.1e}')
        
        # Check for convergence
        if iter > 0 and normChange < tol:
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
    d, m = A.shape
    device = A.device
    
    # Create sparse matrix for sketching
    indices = torch.stack([h.long(), torch.arange(d, device=device)])
    values = s
    S = torch.sparse_coo_tensor(indices, values, (J, d), device=device)
    
    return S @ A

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
        sketches.append(count_sketch(A.T, h, J, s))
    
    # Multiply sketches element-wise
    result = torch.ones_like(sketches[0])
    for sketch in sketches:
        result *= sketch
    
    return result

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

def sparse_tensor_sketch_mat(values, indices, h_list, s_list, J, In, n):
    """Tensor sketch for sparse tensor along mode n."""
    # This would need a custom C++/CUDA implementation for efficiency
    # Placeholder implementation - would be very slow in Python
    warnings.warn("Using slow Python implementation for sparse tensor sketch")
    
    # Get the dimensions
    other_dims = [h.shape[0] for h in h_list]
    other_size = np.prod(other_dims)
    
    # Initialize result
    result = torch.zeros(J, In, device=values.device)
    
    # Iterate through non-zero elements
    for val, idx in zip(values, indices.T):
        # Compute combined hash and sign
        h = 0
        s = 1.0
        stride = 1
        pos = 0
        
        for i in range(len(indices)):
            if i == n:
                in_idx = idx[i]
                continue
            
            dim = h_list[pos].shape[0]
            h = (h + h_list[pos][idx[i]]) % J
            s *= s_list[pos][idx[i]]
            pos += 1
        
        # Update sketch
        result[h, in_idx] += val * s
    
    return result

def sparse_to_sparse_tensor_sketch_mat(values, indices, h_list, s_list, J, In, n):
    """Alternative sparse tensor sketch that returns sparse result."""
    # Placeholder implementation
    warnings.warn("Using slow Python implementation for sparse tensor sketch")
    
    # This would return (indices, values) for sparse matrix
    # Actual implementation would need to be optimized
    pass

def sparse_tensor_sketch_vec(values, indices, h_list, s_list, J):
    """Tensor sketch of sparse tensor as vector."""
    # Placeholder implementation
    warnings.warn("Using slow Python implementation for sparse tensor sketch")
    
    # Compute combined hash and sign functions
    h_combined = torch.zeros(indices.shape[1], dtype=torch.int64, device=values.device)
    s_combined = torch.ones(indices.shape[1], device=values.device)
    
    for i in range(len(h_list)):
        h_combined = (h_combined + h_list[i][indices[i]]) % J
        s_combined *= s_list[i][indices[i]]
    
    # Apply count sketch
    result = torch.zeros(J, device=values.device)
    for h, s, val in zip(h_combined, s_combined, values):
        result[h] += val * s
    
    return result