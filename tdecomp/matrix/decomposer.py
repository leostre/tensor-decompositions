import torch
from torch.ao.quantization.utils import _normalize_kwargs
from functools import wraps, reduce, partial
from typing import *
from abc import ABC, abstractmethod


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

DIM_SUM_LIM = 1024
DIM_LIM = 2048

class Decomposer(ABC):
    def decompose(self, W: torch.Tensor, *args, **kwargs):
        if not self._is_big(W):
            return self._decompose(W, *args, **kwargs)
        else:
            return self._decompose_big(W, *args, **kwargs)
        
    def _is_big(self, W: torch.Tensor):
        return sum(W.size()) > DIM_SUM_LIM or any(d > DIM_LIM for d in W.size())

    @abstractmethod
    def _decompose(self, W, *args, **kwargs):
        pass

    def _decompose_big(self, W, *args, **kwargs):
        return self._decompose(W, *args, **kwargs)
    
    def _get_stable_rank(self, W):
        n_samples = max(W.shape)
        eps = self.distortion_factor
        if eps <= 0 or eps >= 1:
            raise ValueError("distortion_factor must be in (0, 1)")
        min_num_samples = torch.ceil(4 * torch.log(n_samples) / (eps**2 / 2 - eps**3 / 3))
        return min((round(max(min_num_samples)), *W.size(), 1))
    
    def get_approximation_error(self, W, *result_matrices):
        approx = reduce(torch.matmul, result_matrices)
        return torch.linalg.norm(W - approx)


def _need_t(f):
    """Performs matrix transposition for maximal projection effect"""
    @wraps(f)
    def _wrapper(self, W: torch.Tensor, *args, **kwargs):
            m, n = W.size(-2), W.size(-1)
            _is_transposed = m >= n
            weight = W.t() if _is_transposed else W
            tns = f(self, weight, *args, **kwargs)
            return (
                tns if not _is_transposed
                else tuple(t.t() for t in reversed(tns))
            )
    return _wrapper

def _ortho_gen(x: int, y: int):
    P = torch.empty((x, y))
    torch.nn.init.orthogonal_(P)
    return P

def _random_subspace_projection(X, k=None):
    """ 
    W.B. Johnson and J. Lindenstrauss. Extensions of Lipschitz mappings into a Hilbert space. Contemp.
     Math., 26:189 206, 1984
    """
    random_basis = torch.randn(X.shape[1], k, device=X.device)
    Q, _ = torch.linalg.qr(random_basis)
    
    return X @ Q

def _sparse_iid_entries(d, k, s=3):
    """
    Генерирует разреженную проекционную матрицу с элементами {-1, 0, +1} 
    
    Параметры:
        d (int): Исходная размерность.
        k (int): Новая размерность (k << d).
        s (int): Параметр разреженности (по умолчанию 3).
    
    http://www.yaroslavvb.com/papers/achlioptas-database.pdf
    """
    R = torch.randint(0, 2*s, size=(d, k))  
    R = (R == 0).astype(int) - (R == 1).astype(int)  
    return R * torch.sqrt(s)  


def _sparse_jl_matrix(d, k, s=3):
    """
    Генерирует разреженную случайную матрицу проекций с элементами {+1, 0, -1},
    удовлетворяющую Johnson-Lindenstrauss Lemma (JLL) с параметром разреженности s.

    https://eclass.uoa.gr/modules/document/file.php/MATH506/03.%20%CE%98%CE%AD%CE%BC%CE%B1%CF%84%CE%B1%20%CE%B5%CF%81%CE%B3%CE%B1%CF%83%CE%B9%CF%8E%CE%BD/Matousek-VariantsJohnsonLindenstrauss.pdf

    Параметры:
        d (int): Исходная размерность
        k (int): Целевая размерность
        s (int): Параметр разреженности (обычно 1, 2 или 3)
    """
    nnz_indices = torch.randint(0, d, (k, s))
    
    values = (torch.randint(0, 2, (k, s)) * 2 - 1).float()
    
    values *= torch.sqrt(1 / s)
    
    rows = nnz_indices.reshape(-1)
    cols = torch.repeat_interleave(torch.arange(k), s)
    
    R = torch.zeros(d, k)
    R[rows, cols] = values.reshape(-1)
    
    return R


def _four_wise_independent_matrix(d: int, k: int, seed: Optional[int] = None) -> torch.Tensor:
    """
    https://edoliberty.github.io/papers/FastDimensionReduction.pdf
    """
    if not (k & (k - 1) == 0):
        raise ValueError("k must be 2 power for Hadamard matrix")
    
    D = torch.diag(torch.randint(0, 2, (d,)) * 2 - 1).float()
    
    hadamard_size = k
    H = torch.tensor([[1]], dtype=torch.float32)
    while H.size(1) < hadamard_size:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    
    H = H[:d, :k]
    Phi = torch.matmul(D, H)
    Phi = Phi * (1 / torch.sqrt(k))
    
    return Phi.T 


def _lean_walsh_transform(
    d: int, 
    k: int, 
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Генерирует матрицу проекции с использованием Lean Walsh Transform и случайной диагональной матрицы.
    https://edoliberty.github.io/papers/DenseFastRandomProjectionsAndLeanWalshTransforms.pdf

    Параметры:
        d (int): Исходная размерность (количество строк)
        k (int): Целевая размерность (количество столбцов, должна быть степенью 2)
        device (str): Устройство для вычислений ("cpu" или "cuda")
        dtype (torch.dtype): Тип данных тензора
    """
    diag_elements = torch.randint(0, 2, (d,), device=device, dtype=dtype) * 2 - 1
    D = torch.diag(diag_elements)

    eye_k = torch.eye(k, device=device, dtype=dtype)
    n = k
    h = eye_k.clone()
    
    for i in range(int(torch.log2(n))):
        s = 2 ** i
        m = n // s
        h = h.view(-1, m, s)
        half = s // 2
        even = h[..., :half]
        odd = h[..., half:]
        h[..., :half] = even + odd
        h[..., half:] = even - odd
    
    H = h.view_as(eye_k) * (1.0 / torch.sqrt(n))

    if d <= k:
        H = H[:d, :]
    else:
        repeats = (d // k) + 1
        H = torch.cat([H] * repeats, dim=0)[:d, :]

    return torch.matmul(D, H)

def _identity_copies_projection(
    d: int, 
    k: int, 
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """    
    https://edoliberty.github.io/papers/thesis.pdf
    
    Параметры:
        d (int): Исходная размерность
        k (int): Целевая размерность (должна делиться на d)
        device (str): Устройство для вычислений
        dtype (torch.dtype): Тип данных тензора
    """
    copies = k // d
    
    eye = torch.eye(d, device=device, dtype=dtype)
    R = torch.cat([eye] * copies, dim=1)

    perm = torch.randperm(k, device=device)
    R = R[:, perm]
    R *= torch.sqrt(d / k)
    
    return R

class TwoSidedRandomSVD(Decomposer):
    """
    https://scispace.com/pdf/randomized-algorithms-for-computation-of-tucker-1stsnpusvv.pdf
    """
    _random_gens = {
        'normal': lambda x, y : torch.randn(x, y),
        'ortho' : _ortho_gen,
        'random_subspace': _random_subspace_projection,
        'iid_entries': _sparse_iid_entries,
        'sparse_unit_entries': _sparse_jl_matrix,
        'four_wise': _four_wise_independent_matrix,
        'lean_walsh': _lean_walsh_transform,
        'identity_copies': _identity_copies_projection
    }

    def __init__(self, *, distortion_factor: float = 0.6, 
                 random_init: str = 'normal'):
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.distortion_factor = distortion_factor  # Store as class attribute
        self.random_init = random_init
    
    @_need_t
    def _decompose_big(self, X: torch.Tensor, rank: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rank is None:
            rank = self._calculate_rank_from_epsilon(X)
        return self._two_sided_decompose(X, rank=rank)
        
    @_need_t
    def _decompose(self, X: torch.Tensor, rank: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rank is None:
            rank = self._calculate_rank_from_epsilon(X)
        return self._two_sided_decompose(X, rank=rank)
    
    def _calculate_rank_from_epsilon(self, tensor: torch.Tensor) -> int:
        svals = torch.linalg.svdvals(tensor)
        stable_rank = (svals.sum() / svals.max())**2
        # Use class attribute instead of parameter
        return max(1, min(tensor.size(-1), int(stable_rank * (1 / self.distortion_factor))))
    
    @_need_t
    def _two_sided_decompose(self, X: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        I, J = X.shape[-2], X.shape[-1]
        random_gen = self._random_gens[self.random_init]
        Omega1 = random_gen(J, rank).to(X.device, X.dtype)
        Omega2 = random_gen(I, rank).to(X.device, X.dtype)
            
        Y1 = X @ Omega1
        Y2 = X.T @ Omega2
            
        Q1, _ = torch.linalg.qr(Y1, mode='reduced')
        Q2, _ = torch.linalg.qr(Y2, mode='reduced')
            
        B = Q1.T @ X @ Q2
            
        U_bar, S, Vh_bar = torch.linalg.svd(B, full_matrices=False)
        U = Q1 @ U_bar
        Vh = (Q2 @ Vh_bar.T).T  
            
        return U, S, Vh
    
    def _two_sided_compose(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
        return (U * S.unsqueeze(0)) @ Vh
    
    def get_approximation_error(self, W: torch.Tensor, *result_matrices: torch.Tensor) -> torch.Tensor:
        U, S, Vh = result_matrices
        approx = (U * S) @ Vh
        return torch.linalg.norm(W - approx)