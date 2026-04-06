from functools import partial, wraps
from typing import Callable, Optional

import tensorflow as tf
import torch
import tensorly as tl

from torch.ao.quantization.utils import _normalize_kwargs

from tdecomp.types import TensorLike, BOOL_TYPE
__all__ = [
    'filter_kw_universal',
    'conjugate_gradient',
    'svd_solver_tikhonov'
]

def filter_kw_universal(f):
    """Automatically switches between fedot-style and conventional init"""
    @wraps(f)
    def _wrapping(self, *args, **kwargs):
        if (len(args) == 1 and isinstance(args[0], dict) and not len(kwargs)):
            params = args[0]
            args = args[1:]
        elif 'params' in kwargs and len(kwargs) == 1:
            params = kwargs['params']
        else:
            params = kwargs
        new_kw = _normalize_kwargs(f, params)
        f(self, *args, **new_kw)

    return _wrapping

def conjugate_gradient(A, b, precond=None, x0=None, 
                       max_iter=100, tol=1e-6, 
                       verbose=False, 
                       device=None):
    """
    Solves Ax = b using the Conjugate Gradient method.
    
    Args:
        A: Linear operator (callable or matrix). If callable, A(x) should return A @ x.
        b: Right-hand side vector (n,)
        x0: Initial guess (n,). If None, uses zero vector.
        max_iter: Maximum iterations
        tol: Tolerance for residual norm
        verbose: Print progress
        
    Returns:
        x: Solution vector (n,)
        residuals: List of residual norms
    """
    eps_zero = 1e-8
    if not device and not callable(A):
        device = A.device
    if callable(precond):
        precond = precond.to(device)
    # Initialize
    x = torch.zeros_like(b, device=device) if x0 is None else x0.clone()
    # print(A.si b.size(), x.size())
    r = b - (A(x) if callable(A) else A @ x)
    if precond is not None:
        z = precond(r) if callable(precond) else precond @ r
    else:
        z = r
    p = z.clone()
    rs_old = r.dot(z)    
    residuals = [torch.norm(r).item()]
    if verbose:
        print(f"Iter 0: Residual = {residuals[-1]:.3e}")
    
    # CG iterations
    for k in range(1, max_iter + 1):
        Ap = A(p) if callable(A) else A @ p
        assert not torch.isnan(r).any(), f'iter {k}'
        alpha = rs_old / (p.dot(Ap) + eps_zero)
        print(p.dot(Ap), alpha)
        
        x += alpha * p
        r -= alpha * Ap
        assert not torch.isnan(x).any(), f'iter {k}'
        residuals.append(torch.norm(r).item())
        
        if verbose and (k % 10 == 0 or k == max_iter):
            print(f"Iter {k}: Residual = {residuals[-1]:.3e}")
        
        if residuals[-1] < tol:
            break
        if precond is None:
            z = p
        else:
            z = precond(r) if callable(precond) else precond @ r
        rs_new = r.dot(z)     
        beta = rs_new / (rs_old + eps_zero)
        p = z + beta * p
        rs_old = rs_new
    assert not torch.isnan(x).any()
    return x, residuals


def svd_solver_tikhonov(A: TensorLike, b: TensorLike, svd_func: Optional[Callable]=None, tol=1e-6, maxiter=20) -> TensorLike:
    """
    Solve Ax = b
        A is (m x n)
        b is (m)
    Assume m >= n
    """
    if svd_func is None:
        svd_func = partial(tl.truncated_svd, n_eigenvecs=min(tl.shape(A)))
    lmbd = 1e-4 
    lmbd_decay = 0.8
    U, S, Vh = svd_func(A)
    Utb = tl.matmul(tl.transpose(U), b)
    S2 = S * S
    x = tl.zeros(tl.shape(A)[1])
    for _ in range(maxiter):
        Sinv = S / (S2 + lmbd**2)  # Wiener filter
        x = tl.matmul(tl.transpose(Vh), (Sinv * Utb))
        if tl.norm(tl.matmul(A, x) - b, order=2) < tol:
            break 
        lmbd *= lmbd_decay
    return x


def pseudo_inverse(A: TensorLike) -> TensorLike:
    '''Find pseudo inverse <b>matrix</b>. Can fail, if A is singular.'''
    AT = tl.transpose(A)
    return tl.solve(tl.matmul(AT, A), AT)

def randperm(k: int, context: dict = {}) -> TensorLike:
    '''Analogue of randperm in torch for tensorly. Returns random vector of int numbers from [0, k).
    Params:
        context: context from tl.context method. Could contain 'device', 'dtype' and etc.
    '''
    return tl.argsort(tl.random.random_tensor((k,), **context), 0)

def is_complex(X: TensorLike) -> bool:
    '''Backend-independent check whether tensor complex or real. 
    
    WARN! Tested only on `pytorch` and `numpy` backends
    '''
    if "complex" in str(tl.context(X)["dtype"]):
        return True
    return False
    #as examples see numpy.complex128 and torch.complex128

def is_floating_point(X: TensorLike) -> bool:
    '''Backend-independent check whether tensor has float dtype or not.'''
    if "float" in str(tl.context(X)["dtype"]):
        return True
    return False
    #as example torch.float, torch.float32, torch.float64, numpy.float64, numpy.float16, numpy.float32, tensorflow.float16

def no_grad(func):
    '''Wrapped function doesn't save DAG for gradients on tensors in any form (backend-independent) during it's execution context.

    For example in pytorch it force every computational result tensor inside function to have 'requires_grad=False' and not to have grad_fn.
    See `@torch.no_grad()`
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        backend = tl.get_backend()
        if backend == "pytorch": #TODO refactor? on KERAS_BACKEND (os.environ["KERAS_BACKEND"] = "torch")
            with torch.no_grad():
                return func(*args, **kwargs)
        elif backend == "tensorflow":
            with tf.GradientTape() as t:
                with t.stop_recording():
                    return func(*args, **kwargs)
        elif backend == "numpy":
            return func(*args, **kwargs)
        else:
            raise NotImplementedError(f"Backend `{backend}` haven't support of @no_grad yet!")
    
    return wrapper

def topk_ids(x: TensorLike, k: int) -> TensorLike:
    '''Returns indices of topk elements in ascending order of elements'''
    return tl.argsort(x, 0)[-k:]

def bool_mask(shape: int | tuple[int], context: dict = {}) -> TensorLike:
    context["dtype"] = BOOL_TYPE
    return tl.zeros(shape, **context)

def numel(x: TensorLike) -> TensorLike:
    '''
    Returns:
        tensor: permutation of all x shapes packed in tensor
    '''
    return tl.prod(tl.tensor(tl.shape(x)))


def multinomial(weights: TensorLike, k: int, context={}) -> TensorLike:
    """
    Efraimidis–Spirakis algorithm (A-Res) for weighted sampling without replacement (replacement=False).
    For each element with weight w_i: key_i = u_i^(1/w_i) where u_i ~ Uniform(0,1).
    Select k elements (indexes) with largest keys.
    
    Args:
        weights: 1D tensorly tensor of non-negative weights (<b>not necessarily normalized</b>).
        k: number of samples.
        context: tensorly context dict.
    """

    # Efraimidis–Spirakis: key_i = u_i^(1/w_i) or the same as exp(log(u_i^(1/w_i))) -> log(u_i)/w_i
    eps = 1e-12
    u = tl.random.random_tensor(tl.shape(weights), **context)
    u = tl.log2(u) / (weights + eps)
    # Select k elements with largest keys
    return tl.argsort(u, 0)[-k:]

def svdvals(x: TensorLike) -> TensorLike:
    '''Returns list of singular values'''
    if tl.get_backend() == "pytorch":
        return torch.linalg.svdvals(x)
    else:
        #slover version via full svd
        return tl.truncated_svd(x, n_eigenvecs=min(tl.shape(x)))[1]