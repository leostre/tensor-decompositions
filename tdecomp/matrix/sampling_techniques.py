import torch

def adaptive_sampling(A: torch.Tensor, k: int, s: int, n_iter: int = 3) -> torch.Tensor:
    """
    Улучшенная версия adaptive_sampling (https://arxiv.org/pdf/0708.3696) с явным решением LS задач
    Args:
        A: входная матрица (m x n)
        k: число столбцов для первой выборки
        s: число дополнительных столбцов
        n_iter: количество итераций уточнения
    Returns:
        C: матрица выбранных столбцов (m x (k+s))
    """
    initial_cols = torch.randperm(A.size(1))[:k]
    C = A[:, initial_cols]
    
    for _ in range(n_iter):
        X = torch.linalg.lstsq(C, A).solution
        
        residual = A - C @ X
        
        probs = torch.norm(residual, p=2, dim=0)**2
        probs /= probs.sum()
        
        extra_cols = torch.multinomial(probs, s, replacement=False)
        C = torch.cat([C, A[:, extra_cols]], dim=1)
    
    return C


def column_select_ls(A: torch.Tensor, k: int, epsilon: float = 0.1) -> torch.Tensor:
    """
    Модифицированный ColumnSelect (file:///C:/Users/user/Downloads/Telegram%20Desktop/Randomized_Algorithms_for_Computation_of_Tucker_Decomposition_and.pdf)
    Args:
        A: Входная матрица (m x n)
        k: Целевой ранг аппроксимации
        epsilon: Параметр точности
    Returns:
        C: Матрица выбранных столбцов (m x c)
    """
    m, n = A.shape
    
    Q, R = torch.linalg.qr(A)
    U = Q[:, :k]  
    
    leverage_scores = torch.norm(U, dim=1)**2
    c = int(k * torch.log(torch.tensor(k)) / epsilon**2)
    c = min(c, n)
    
    probs = leverage_scores / k
    selected_cols = torch.multinomial(probs, c, replacement=True)
    
    C = A[:, selected_cols]
    scales = torch.linalg.lstsq(C, A).solution.norm(dim=1)
    C = C * scales.reshape(1, -1)
    
    return C