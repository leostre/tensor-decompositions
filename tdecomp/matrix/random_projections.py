from enum import Enum
from functools import partial, partialmethod
import math
from typing import *
import tensorly as tl
import tdecomp
from tdecomp.types import TensorLike

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
    '''Generates tensor from normal distribution.

    WARN: Can lie in positive way on many iterations!
    '''
    return tl.randn((rows, cols), **context)


def ortho(rows: int, cols: int, context: dict = _default_context) -> TensorLike:
    P = None
    if (rows >= cols):
        P = normal(rows, cols, context)
    else:
        P = normal(cols, rows, context)
        
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


def sparse_jl_matrix(d: int, k: int, s: int = 3, context: dict = _default_context) -> TensorLike:
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
            cols = tl.index_update(cols, tl.index[s * i + j], i)
    
    R = tl.zeros((d, k))
    R = tl.index_update(R, tl.index[rows, cols], tl.reshape(values, (-1,)))
    
    return R


def four_wise_independent_matrix(d: int, k: int, context: dict = _default_context) -> TensorLike:
    """
    https://edoliberty.github.io/papers/FastDimensionReduction.pdf

    Args:
        k: power of 2
    """
    if not (k > 0 and (k & (k - 1) == 0)):
        raise ValueError("k must be 2 power for Hadamard matrix")
    
    D = tl.diag(tl.random.random_tensor((d,), **context) * 4 // 1 - 1)
    
    hadamard_size = k
    H = tl.tensor([[1]], **context)
    while tl.shape(H)[1] < hadamard_size:
        H = tl.concatenate([
            tl.concatenate([H, H], axis=1),
            tl.concatenate([H, -H], axis=1)
        ], axis=0)
    
    H = H[:d, :k]
    Phi = tl.matmul(D, H)
    Phi = Phi * (1 / math.sqrt(k))
    
    return Phi 


def lean_walsh(d: int, k: int, context: dict = _default_context) -> TensorLike:
    """
    Генерирует матрицу проекции с использованием Lean Walsh Transform и случайной диагональной матрицы.
    https://edoliberty.github.io/papers/DenseFastRandomProjectionsAndLeanWalshTransforms.pdf

    Параметры:
        d (int): Исходная размерность (количество строк)
        k (int): Целевая размерность (количество столбцов, должна быть степенью 2)
    """
    if not (k > 0 and (k & (k - 1) == 0)):
        raise ValueError("k must be a power of 2")

    diag_elements = tl.random.random_tensor((d,), **context) * 4 // 1 - 1
    D = tl.diag(diag_elements)

    eye_k = tl.eye(k, **context)
    h = tl.tensor(eye_k, **context)
    
    num_iterations = int(math.log2(k))
    
    for i in range(num_iterations):
        s = 2 ** i
        m = k // s
        h = tl.reshape(h, (-1, m, s))
        half = s // 2
        if half == 0:
            break
        even = h[..., :half]
        odd = h[..., half:]
        h = tl.index_update(h, tl.index[..., :half], even + odd)
        h = tl.index_update(h, tl.index[..., half:], even - odd)
    
    h = tl.reshape(h, (k, k))
    H = h * (1.0 / math.sqrt(k))

    if d <= k:
        H = H[:d, :]
    else:
        repeats = (d // k) + 1
        H = tl.concatenate([H] * repeats, axis=0)[:d, :]

    return tl.matmul(D, H)

def identity_copies(d: int, k: int, context: dict = _default_context) -> TensorLike:
    """    
    https://edoliberty.github.io/papers/thesis.pdf
    
    Параметры:
        d (int): Исходная размерность
        k (int): Целевая размерность (должна делиться на d)
    """
    copies = k // d
    remainder = k % d
    
    eye = tl.eye(d, **context)
    R_parts = [eye] * copies
    
    if remainder > 0:
        R_parts.append(eye[:, :remainder])
    
    R = tl.concatenate(R_parts, axis=1)

    perm = tdecomp.utils.randperm(k, context)
    R = R[:, perm]
    R *= math.sqrt(d / k)
    
    return R

class ProjectorGenerator(Enum):
    normal = partial(normal)
    ortho = partial(ortho)
    sparse_iid_entries = partial(sparse_iid_entries)
    sparse_jl_matrix = partial(sparse_jl_matrix)
    four_wise_independent_matrix = partial(four_wise_independent_matrix)
    lean_walsh = partial(lean_walsh)
    identity_copies = partial(identity_copies)

class Projector:
    def __init__(self, mode: ProjectorGenerator):
        self.P = None
        self.mode = mode

    def generate_P(self, d: int, k: int, context: dict, **generator_kws) -> TensorLike:
        P = self.mode.value(d, k, context, **generator_kws)
        return P
    
    def project(self, tensor: TensorLike, proj_dim: int, *, side: Literal['left', 'right'], renew: bool = True, **gen_kws):
        d = tl.shape(tensor)[0 if side == 'left' else -1]
        if (self.P is None) or renew or (tl.shape(self.P)[0] != proj_dim) or (tl.shape(self.P)[-1] != d):
            self.P = self.generate_P(proj_dim, d, tl.context(tensor), **gen_kws)
        if side == 'left':
            return tl.matmul(self.P, tensor)
        else:
            return tl.matmul(tensor, tl.transpose(self.P))
    
    lproject = partialmethod(project, side='left')
    rproject = partialmethod(project, side='right')