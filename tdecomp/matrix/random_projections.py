from functools import partialmethod, reduce
from typing import *

import torch


__all__ = [
    'normal',
    'ortho',
    'random_subspace_projection_gen',
    'sparse_iid_entries',
    'sparse_jl_matrix',
    'four_wise_independent_matrix',
    'lean_walsh',
    'identity_copies',
    'Projector'
]


def normal(x: int, y: int, device: str = 'cpu', dtype=torch.float32):
    return torch.randn(x, y, device=device, dtype=dtype)


def ortho(x: int, y: int, device: str = 'cpu', dtype=torch.float32):
    P = torch.empty((x, y), device=device, dtype=dtype)
    torch.nn.init.orthogonal_(P)
    return P


def sparse_iid_entries(d, k, s=3, device: str = 'cpu', dtype=torch.float32):
    """
    Генерирует разреженную проекционную матрицу с элементами {-1, 0, +1} 
    
    Параметры:
        d (int): Исходная размерность.
        k (int): Новая размерность (k << d).
        s (int): Параметр разреженности (по умолчанию 3).
    
    http://www.yaroslavvb.com/papers/achlioptas-database.pdf
    """
    R = torch.randint(0, 2*s, size=(d, k), device=device, dtype=dtype)  
    R = (R == 0).to(torch.int) - (R == 1).to(torch.int) 
    return R * torch.sqrt(torch.tensor(s, dtype=torch.float32))  


def sparse_jl_matrix(d, k, s=3, device: str = 'cpu', dtype=torch.float32):
    """
    Генерирует разреженную случайную матрицу проекций с элементами {+1, 0, -1},
    удовлетворяющую Johnson-Lindenstrauss Lemma (JLL) с параметром разреженности s.

    https://eclass.uoa.gr/modules/document/file.php/MATH506/03.%20%CE%98%CE%AD%CE%BC%CE%B1%CF%84%CE%B1%20%CE%B5%CF%81%CE%B3%CE%B1%CF%83%CE%B9%CF%8E%CE%BD/Matousek-VariantsJohnsonLindenstrauss.pdf

    Параметры:
        d (int): Исходная размерность
        k (int): Целевая размерность
        s (int): Параметр разреженности (обычно 1, 2 или 3)
    """
    nnz_indices = torch.randint(0, d, (k, s), device=device, dtype=dtype)
    
    values = (torch.randint(0, 2, (k, s), device=device, dtype=dtype) * 2 - 1).float()
    
    values *= torch.sqrt(1 / s)
    
    rows = nnz_indices.reshape(-1)
    cols = torch.repeat_interleave(torch.arange(k), s)
    
    R = torch.zeros(d, k)
    R[rows, cols] = values.reshape(-1)
    
    return R


def four_wise_independent_matrix(d: int, k: int, 
                                 device: str = "cpu",
                                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    https://edoliberty.github.io/papers/FastDimensionReduction.pdf
    """
    if not (k & (k - 1) == 0):
        raise ValueError("k must be 2 power for Hadamard matrix")
    
    D = torch.diag(torch.randint(0, 2, (d,), device=device, dtype=dtype) * 2 - 1).float()
    
    hadamard_size = k
    H = torch.tensor([[1]], device=device, dtype=dtype)
    while H.size(1) < hadamard_size:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    
    H = H[:d, :k]
    Phi = torch.matmul(D, H)
    Phi = Phi * (1 / torch.sqrt(k))
    
    return Phi.T 


def lean_walsh(
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
    if not (k > 0 and (k & (k - 1) == 0)):
        raise ValueError("k must be a power of 2")

    diag_elements = torch.randint(0, 2, (d,), device=device, dtype=dtype) * 2 - 1
    D = torch.diag(diag_elements)

    eye_k = torch.eye(k, device=device, dtype=dtype)
    h = eye_k.clone()
    
    num_iterations = int(torch.log2(torch.tensor(k, dtype=dtype, device=device)))
    
    for i in range(num_iterations):
        s = 2 ** i
        m = k // s
        h = h.view(-1, m, s)
        half = s // 2
        if half == 0:
            break
        even = h[..., :half]
        odd = h[..., half:]
        h[..., :half] = even + odd
        h[..., half:] = even - odd
    
    H = h.view_as(eye_k) * (1.0 / torch.sqrt(torch.tensor(k, dtype=torch.float32)))

    if d <= k:
        H = H[:d, :]
    else:
        repeats = (d // k) + 1
        H = torch.cat([H] * repeats, dim=0)[:d, :]

    return torch.matmul(D, H)

def identity_copies(
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
    remainder = k % d
    
    eye = torch.eye(d, device=device, dtype=dtype)
    R_parts = [eye] * copies
    
    if remainder > 0:
        R_parts.append(eye[:, :remainder])
    
    R = torch.cat(R_parts, dim=1)

    perm = torch.randperm(k, device=device)
    R = R[:, perm]
    R *= torch.sqrt(torch.tensor(d / k, dtype=torch.float32))
    
    return R


class Projector:
    def __init__(self, mode: str):
        self.P = None
        self.mode = mode

    def generate_P(self, d: int, k: int, **generator_kws) -> torch.Tensor:
        P = RANDOM_GENS[self.mode](d, k, **generator_kws)
        return P
    
    def project(self, tensor: torch.Tensor, proj_dim: int, *, side: Literal['left', 'right'], renew: bool = True, **gen_kws):
        d = tensor.size(0 if side == 'left' else -1)
        if self.P is None:
            self.P = self.generate_P(d, proj_dim, device=tensor.device, **gen_kws)
        matrices = [self.P, tensor]
        if renew:
            self.P = self.generate_P(d, proj_dim,  device=tensor.device, **gen_kws)
        if side == 'right':
            self.P = self.P
            matrices = reversed(matrices)
        projected = reduce(torch.matmul, matrices)
        return projected
    
    lproject = partialmethod(project, side='left')
    rproject = partialmethod(project, side='right')


__locals = locals()
RANDOM_GENS = {
    name: func for name, func in __locals.items() if not name in ('Projector',)
}