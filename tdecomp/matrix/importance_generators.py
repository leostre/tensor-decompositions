from enum import Enum
from functools import partial
from typing import *

import tensorly as tl
from tdecomp.types import TensorLike

__all__ = [
    'l1_norm',
    'l2_norm',
    'linf_norm',
    'fro_norm',
    'ridge_leverage',
    'ImportanceComputer'
]

def _normalize_importances(col_norms: TensorLike, row_norms: TensorLike) -> tuple[TensorLike, TensorLike]:
    return col_norms / (tl.sum(col_norms) + 1e-10), row_norms / (tl.sum(row_norms) + 1e-10)

def l1_norm(X: TensorLike) -> tuple[TensorLike, TensorLike]:
    col_norms = tl.norm(X, order=1, axis=0) 
    row_norms = tl.norm(X, order=1, axis=1)    
    return _normalize_importances(col_norms, row_norms)

def l2_norm(X: TensorLike) -> tuple[TensorLike, TensorLike]:
    col_norms = tl.norm(X, order=2, axis=0) 
    row_norms = tl.norm(X, order=2, axis=1)
    return _normalize_importances(col_norms, row_norms)

def linf_norm(X: TensorLike) -> tuple[TensorLike, TensorLike]:
    col_norms = tl.norm(X, order=float('inf'), axis=0)
    row_norms = tl.norm(X, order=float('inf'), axis=1)
    return _normalize_importances(col_norms, row_norms)

def fro_norm(X: TensorLike) -> tuple[TensorLike, TensorLike]:
    '''Diffs from l2_norm in squared sum'''
    x_squared = X * X
    col_scores = (x_squared).sum(dim=0)
    row_scores = (x_squared).sum(dim=1)
    return _normalize_importances(col_scores, row_scores)

def ridge_leverage(
    X: TensorLike,
    lam: Optional[float] = None,
) -> Tuple[TensorLike, TensorLike]:
    m, n = tl.shape(X)
    
    if lam is None:
        lam = 1e-6 * (tl.sum(X * X) / max(m, n))
    

    Xt = tl.transpose(X)
    if m >= n:
        # Случай "Длинная матрица": инвертируем n x n
        XtX = tl.matmul(Xt, X)
        I = tl.eye(n, **tl.context(X))
        M_reg = XtX + lam * I # type: ignore
        M_inv = tl.solve(M_reg, I) # n x n

        # 1. Row Scores: diag(X M_inv X^T) -> построчно x_i M_inv x_i^T (m x m)
        XM = tl.matmul(X, M_inv)
        row_scores = tl.sum(XM * X, axis=1)
        
        # 2. Col Scores: diag( 1/lam * (G - G M_inv G) )
        # G = XtX. Считаем K = G @ M_inv @ G
        term2 = tl.matmul(XtX, tl.matmul(M_inv, XtX))
        col_scores = (tl.diag(XtX) - tl.diag(term2)) / lam
        
    else:
        # Случай "Широкая матрица": инвертируем m x m
        XXt = tl.matmul(X, Xt)
        I = tl.eye(m, **tl.context(X))
        M_reg = XXt + lam * I
        M_inv = tl.solve(M_reg, I) # m x m
        
        # 1. Col Scores: diag(X^T M_inv X)
        XtM = tl.matmul(Xt, M_inv) # n x m
        col_scores = tl.sum(XtM * Xt, axis=1)
        
        # 2. Row Scores: diag( 1/lam * (G - G M_inv G) ) где G = XXt
        term2 = tl.matmul(XXt, tl.matmul(M_inv, XXt))
        row_scores = (tl.diag(XXt) - tl.diag(term2)) / lam

    # Clamp для удаления численного шума (например -1e-16)
    row_scores = tl.clip(row_scores, 0.0, None)
    col_scores = tl.clip(col_scores, 0.0, None)
    
    return _normalize_importances(col_scores, row_scores)


class ColumnRowImportancesGenerator(Enum):
    l1_norm = partial(l1_norm)
    l2_norm = partial(l2_norm)
    linf_norm = partial(linf_norm)
    fro_norm = partial(fro_norm)
    ridge_leverage = partial(ridge_leverage)

class ImportanceComputer:

    def __init__(self, mode: ColumnRowImportancesGenerator):
        self.mode = mode
    
    def compute(self, X: TensorLike) -> Tuple[TensorLike, TensorLike]:
        method = self.mode.value
        return method(X)


__locals = locals()
IMPORTANCE_GENS = {
    name: func for name, func in __locals.items() if name not in ('ImportanceComputer',)
}