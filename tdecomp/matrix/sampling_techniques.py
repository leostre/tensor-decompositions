import torch
from tdecomp._base import BaseSketch, _need_t

__all__ = [
    'adaptive_sampling',
    'column_select'
]

class AdaptiveSamplingSketch(BaseSketch):
    """
    Улучшенная версия adaptive_sampling (https://arxiv.org/pdf/0708.3696) с явным решением LS задач
    Args:
        sketch_size: Желаемый размер скетча (абсолютный или относительный)
        compression_ratio: Коэффициент сжатия для автоматического определения размера скетча
        power_iterations: Количество степенных итераций для оценки важности столбцов
                         (больше итераций → точнее оценка, но медленнее)
    Returns:
        Матрица скетча 
    """
    def __init__(self, sketch_size: int | None = None, 
                 compression_ratio: float = 0.5,
                 power_iterations: int = 2):
        super().__init__(sketch_size, compression_ratio, 'adaptive')
        self.power_iterations = power_iterations

    def _estimate_column_importance(self, matrix: torch.Tensor) -> torch.Tensor:
        n_cols = matrix.size(1)
        v = torch.randn(n_cols, 1, device=matrix.device, dtype=matrix.dtype)
        
        for _ in range(self.power_iterations):
            v = matrix.T @ (matrix @ v)
            v /= torch.norm(v)
        
        importance = torch.abs(matrix @ v).squeeze()
        return importance / importance.sum()

    def _sketch(self, matrix: torch.Tensor, sketch_size: int) -> torch.Tensor:
        m, n = matrix.shape
        col_importance = self._estimate_column_importance(matrix)
        selected_cols = torch.multinomial(col_importance, sketch_size, replacement=False)
        
        return matrix[:, selected_cols]


class ColumnSelectSketch(BaseSketch):
    """
     Модифицированный ColumnSelect 
    Args:
        sketch_size: Размер скетча
        compression_ratio: Коэффициент сжатия
        epsilon: Параметр точности
        power_iterations: Количество степенных итераций для оценки leverage scores
    Returns:
        Матрица скетча
    """
    def __init__(self, sketch_size: int | None = None,
                 compression_ratio: float = 0.5,
                 epsilon: float = 0.1,
                 power_iterations: int = 2):
        super().__init__(sketch_size, compression_ratio, 'leverage')
        self.epsilon = epsilon
        self.power_iterations = power_iterations

    def _fast_leverage_scores(self, matrix: torch.Tensor, k: int) -> torch.Tensor:
        m, n = matrix.shape
        random_matrix = torch.randn(n, k, device=matrix.device, dtype=matrix.dtype)
        projected = matrix @ random_matrix
        Q, _ = torch.linalg.qr(projected, mode='reduced')
        leverage_scores = torch.norm(Q, dim=1)**2
        return leverage_scores / leverage_scores.sum()

    def _sketch(self, matrix: torch.Tensor, sketch_size: int) -> torch.Tensor:
        m, n = matrix.shape
        
        k_approx = min(10, sketch_size)
        leverage_scores = self._fast_leverage_scores(matrix, k_approx)
        
        c = int(sketch_size * torch.log(torch.tensor(sketch_size + 1)) / self.epsilon**2)
        c = min(c, n, sketch_size)
        
        selected_cols = torch.multinomial(leverage_scores, c, replacement=True)
        return matrix[:, selected_cols]


SAMPLING_TECHNIQUES = {
    'adaptive_sampling': AdaptiveSamplingSketch,
    'column_select': ColumnSelectSketch
}