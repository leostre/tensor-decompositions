import copy
from enum import Enum
from functools import partial, partialmethod, reduce
import math
from typing import *
import tensorly as tl

import torch

from tdecomp._base import TensorLike


__all__ = [
    'normal',
    'ortho',
    'sparse_iid_entries',
    'sparse_jl_matrix',
    'four_wise_independent_matrix',
    'lean_walsh',
    'identity_copies',
    'Projector'
]


_default_context = {"device": "cpu", "dtype": tl.float32}
"""Default tensor context for creation"""

def normal(rows: int, cols: int, context: dict = _default_context) -> TensorLike:
    return tl.randn((rows, cols), **context)


def ortho(rows: int, cols: int, context: dict = _default_context) -> TensorLike:
    P = normal(rows, cols, context)
    if (rows < cols):
        P = tl.transpose(P)

    q, r = tl.qr(P, mode="reduced")
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    ph = tl.sign(tl.diag(r))
    q = tl.einsum("ij,j->ij", q, ph)

    if (rows < cols):
        q = tl.transpose(q)
    return q


def sparse_iid_entries(d: int, k: int, s: int = 3, context: dict = _default_context) -> TensorLike:
    """
    Генерирует разреженную проекционную матрицу с элементами {-1, 0, +1} 
    
    Параметры:
        d (int): Исходная размерность.
        k (int): Новая размерность (k << d).
        s (int): Параметр разреженности (по умолчанию 3).
    
    http://www.yaroslavvb.com/papers/achlioptas-database.pdf
    """
    R = tl.random.random_tensor((d, k), **context) * 2 * s // 1 #floor to int
    R = (R == 0) * 1 - (R == 1) * 1 #cast to int
    return R * math.sqrt(s) 


def sparse_jl_matrix(d: int, k: int, s: int = 3, context: dict = _default_context):
    """
    Генерирует разреженную случайную матрицу проекций с элементами {+1, 0, -1},
    удовлетворяющую Johnson-Lindenstrauss Lemma (JLL) с параметром разреженности s.

    https://eclass.uoa.gr/modules/document/file.php/MATH506/03.%20%CE%98%CE%AD%CE%BC%CE%B1%CF%84%CE%B1%20%CE%B5%CF%81%CE%B3%CE%B1%CF%83%CE%B9%CF%8E%CE%BD/Matousek-VariantsJohnsonLindenstrauss.pdf

    Параметры:
        d (int): Исходная размерность
        k (int): Целевая размерность
        s (int): Параметр разреженности (обычно 1, 2 или 3)
    """
    cols = tl.tensor([0] * (k * s))
    int_dtype = cols.dtype

    nnz_indices = tl.tensor(tl.random.random_tensor((k, s), **context) * d, dtype=int_dtype)
    
    values = tl.random.random_tensor((k, s), **context) * 4 // 1 - 1
    
    values *= math.sqrt(1 / s)
    
    rows = tl.reshape(nnz_indices, (-1,)) # "," here is important!

    #cols = torch.repeat_interleave(tl.arange(k), s) - analogue
    for i in range(1, k):
        for j in range(s):
            tl.index_update(cols, tl.index[s * i + j], i)
    
    R = tl.zeros((d, k))
    tl.index_update(R, tl.index[rows, cols], tl.reshape(values, (-1,)))
    
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
            self.P = self.generate_P(proj_dim, d , device=tensor.device, **gen_kws)
        matrices = [self.P, tensor] #TODO fix
        if renew:
            self.P = self.generate_P(d, proj_dim,  device=tensor.device, **gen_kws)
        if side == 'right':
            self.P = self.P
            matrices = reversed(matrices)
        projected = reduce(torch.matmul, matrices)
        return projected
    
    lproject = partialmethod(project, side='left')
    rproject = partialmethod(project, side='right')


class PROJECTOR_GENS(Enum):
    normal: Callable[[int, int], TensorLike] = partial(normal)
    ortho = partial(ortho)
    sparse_iid_entries = partial(sparse_iid_entries)
    sparse_jl_matrix = partial(sparse_jl_matrix)
    four_wise_independent_matrix = partial(four_wise_independent_matrix)
    lean_walsh = partial(lean_walsh)
    identity_copies = partial(identity_copies)