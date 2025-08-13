import torch

__all__ = [
    'adaptive_sampling',
    'column_select'
]

def adaptive_sampling_inv(X: torch.Tensor, k: int, s: int, n_iter: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Улучшенная версия adaptive_sampling (https://arxiv.org/pdf/0708.3696) с явным решением LS задач
    Args:
        X: входная матрица (m x n)
        k: число столбцов для первой выборки
        s: число дополнительных столбцов
        n_iter: количество итераций уточнения
    Returns:
        S, B
    """
    X_pinv = torch.linalg.pinv(X)  
    initial_cols = torch.randperm(X_pinv.size(1))[:k]
    S = X_pinv[:, initial_cols]
    
    for _ in range(n_iter):
        B = torch.linalg.lstsq(S, X_pinv).solution
        residual = X_pinv - S @ B
        probs = torch.norm(residual, p=2, dim=0)**2
        probs /= probs.sum()
        extra_cols = torch.multinomial(probs, s, replacement=False)
        S = torch.cat([S, X_pinv[:, extra_cols]], dim=1)
    
    B = torch.linalg.lstsq(S, X_pinv).solution
    return S, B


def column_select_inv(X: torch.Tensor, k: int, epsilon: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Модифицированный ColumnSelect (file:///C:/Users/user/Downloads/Telegram%20Desktop/Randomized_Algorithms_for_Computation_of_Tucker_Decomposition_and.pdf)
    Args:
        A: Входная матрица (m x n)
        k: Целевой ранг аппроксимации
        epsilon: Параметр точности
    Returns:
        S, B
    """
    X_pinv = torch.linalg.pinv(X)
    m, n = X_pinv.shape
    Q, _ = torch.linalg.qr(X_pinv)
    U = Q[:, :k]  
    leverage_scores = torch.norm(U, dim=1)**2
    c = int(k * torch.log(torch.tensor(k)) / epsilon**2)
    c = min(c, n)
    probs = leverage_scores / k
    selected_cols = torch.multinomial(probs, c, replacement=True)
    S = X_pinv[:, selected_cols]
    B = torch.linalg.lstsq(S, X_pinv).solution
    return S, B

SAMPLING_TECHNIQUES = {
        'adaptive_sampling' : adaptive_sampling_inv,
        'column_select' : column_select_inv
    }