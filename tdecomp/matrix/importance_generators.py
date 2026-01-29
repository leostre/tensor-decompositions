from functools import partialmethod, reduce
from typing import *
import torch
import math


__all__ = [
    'l1_norm',
    'l2_norm',
    'linf_norm',
    'fro_norm',
    'ridge_leverage',
    'ImportanceComputer'
]

def l1_norm(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    col_norms = torch.linalg.norm(X, ord=1, dim=0) 
    row_norms = torch.linalg.norm(X, ord=1, dim=1)
    
    col_probs = col_norms / (col_norms.sum() + 1e-10)
    row_probs = row_norms / (row_norms.sum() + 1e-10)
    
    return col_probs, row_probs

def l2_norm(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    col_norms = torch.linalg.norm(X, ord=2, dim=0) 
    row_norms = torch.linalg.norm(X, ord=2, dim=1)
    
    col_probs = col_norms / (col_norms.sum() + 1e-10)
    row_probs = row_norms / (row_norms.sum() + 1e-10)
    
    return col_probs, row_probs

def linf_norm(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    col_scores = torch.linalg.norm(X, ord=float('inf'), dim=0)
    row_scores = torch.linalg.norm(X, ord=float('inf'), dim=1)

    col_probs = col_scores / (col_scores.sum() + 1e-10)
    row_probs = row_scores / (row_scores.sum() + 1e-10)

    return col_probs, row_probs

def fro_norm(X: torch.Tensor, eps: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:

    col_scores = (X * X).sum(dim=0)
    row_scores = (X * X).sum(dim=1)

    col_probs = col_scores / (col_scores.sum() + eps)
    row_probs = row_scores / (row_scores.sum() + eps)
    return col_probs, row_probs

def ridge_leverage(
    X: torch.Tensor,
    lam: Optional[float] = None,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = X.shape
    device = X.device
    
    if lam is None:
        lam = 1e-6 * (X.pow(2).sum().item() / max(m, n))
    

    if m >= n:
        # Случай "Длинная матрица": инвертируем n x n
        XtX = X.T @ X
        M_reg = XtX + lam * torch.eye(n, device=device)
        # M_inv = (X^T X + lam I)^-1
        # Используем cholesky_solve для SPD матрицы (быстрее и стабильнее общего solve)
        L = torch.linalg.cholesky(M_reg)
        M_inv = torch.cholesky_solve(torch.eye(n, device=device), L)
        
        # 1. Row Scores: diag(X M_inv X^T) -> построчно x_i M_inv x_i^T
        # B = M_inv @ X.T -> но эффективнее считать (X @ M_inv) * X
        XM = X @ M_inv # m x n
        row_scores = (XM * X).sum(dim=1)
        
        # 2. Col Scores: diag( 1/lam * (G - G M_inv G) )
        # G = XtX. Считаем K = G @ M_inv @ G
        # Col scores = (G_jj - K_jj) / lam
        term2 = XtX @ M_inv @ XtX
        col_scores = (torch.diagonal(XtX) - torch.diagonal(term2)) / lam
        
    else:
        # Случай "Широкая матрица": инвертируем m x m
        XXt = X @ X.T
        M_reg = XXt + lam * torch.eye(m, device=device)
        
        L = torch.linalg.cholesky(M_reg)
        M_inv = torch.cholesky_solve(torch.eye(m, device=device), L)
        
        # 1. Col Scores: diag(X^T M_inv X)
        XtM = X.T @ M_inv # n x m
        col_scores = (XtM * X.T).sum(dim=1)
        
        # 2. Row Scores: diag( 1/lam * (G - G M_inv G) ) где G = XXt
        term2 = XXt @ M_inv @ XXt
        row_scores = (torch.diagonal(XXt) - torch.diagonal(term2)) / lam

    # Clamp для удаления численного шума (например -1e-16)
    row_scores = row_scores.clamp_min(0.0)
    col_scores = col_scores.clamp_min(0.0)

    col_probs = col_scores / (col_scores.sum() + eps)
    row_probs = row_scores / (row_scores.sum() + eps)
    
    return col_probs, row_probs


class ImportanceComputer:

    def __init__(self, mode: str):
        self.mode = mode
        if mode not in IMPORTANCE_GENS:
            raise ValueError(f"Unknown importance method: {mode}. "
                           f"Available: {list(IMPORTANCE_GENS.keys())}")
    
    def compute(self, X: torch.Tensor, rank: int = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        method = IMPORTANCE_GENS[self.mode]
        return method(X, rank=rank, **kwargs)
    
    def get_available_methods(self) -> List[str]:
        return list(IMPORTANCE_GENS.keys())


__locals = locals()
IMPORTANCE_GENS = {
    name: func for name, func in __locals.items() if name not in ('ImportanceComputer',)
}