import torch

from functools import wraps, reduce, partial
from typing import *
from abc import ABC, abstractmethod

from tdecomp.matrix.random_projections import RANDOM_GENS


DIM_SUM_LIM = 1024
DIM_LIM = 2048

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

class Decomposer(ABC):
    def __init__(self, rank: Union[int, float] = None, distortion_factor: float = 0.6, 
                 random_init: str = 'normal'):
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.distortion_factor = distortion_factor
        self.random_init = random_init
        self.rank = rank

    def decompose(self, W: torch.Tensor, rank=None, *args, **kwargs):
        if rank is None:
            rank = self.evaluate_parameter(W, 'stable_rank', method='jhonson-lin')
        elif isinstance(rank, float):
            rank = max(1, int(rank * min(W.size())))
        if not self._is_big(W):
            return self._decompose(W, rank, *args, **kwargs)
        else:
            return self._decompose_big(W, rank, *args, **kwargs)
        
    def _is_big(self, W: torch.Tensor):
        return sum(W.size()) > DIM_SUM_LIM or any(d > DIM_LIM for d in W.size())
    
    def _find_optimal_rank(self, X: torch.Tensor, max_rank: Optional[int] = None) -> int:
        if max_rank is None:
            max_rank = min(X.shape)
        
        low, high = 1, max_rank
        best_rank = max_rank

        while low <= high:
            mid = (low + high) // 2
            error = self._decomposition_error(X, mid)

            if error <= self.tolerance:
                best_rank = mid
                high = mid - 1
            else:
                low = mid + 1

        return best_rank
    
    @abstractmethod
    def _decomposition_error(self, X: torch.Tensor, rank: int) -> float:
        pass

    @abstractmethod
    def _decompose(self, W, rank, *args, **kwargs):
        pass
    
    def _decompose_big(self, W, rank, *args, **kwargs):
        return self._decompose(W, rank, *args, **kwargs)
    
    def evaluate_parameter(self, W: torch.Tensor, param_type: str, method: str = None) -> int:
        m, n = W.size(-2), W.size(-1)
        min_dim = min(m, n)
        
        if param_type == 'stable_rank':
            eps = self.distortion_factor
            n_samples = max(m, n)
            min_num_samples = torch.ceil(4 * torch.log(torch.tensor(n_samples)) / (eps**2 / 2 - eps**3 / 3))
            return min(int(min_num_samples.item()), m, n, 1)
        elif param_type == 'sketch_size':
            if hasattr(self, 'sketch_size') and self.sketch_size is not None:
                if isinstance(self.sketch_size, int):
                    return min(self.sketch_size, min_dim)
                else:
                    return max(1, int(self.sketch_size * min_dim))
            elif hasattr(self, 'compression_ratio'):
                return max(1, int(min_dim * self.compression_ratio))
            else:
                return min_dim
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    
    def get_approximation_error(self, W, *result_matrices):
        approx = reduce(torch.matmul, result_matrices)
        return torch.linalg.norm(W - approx)
    
    def compose(self, *factors, **kwargs) -> torch.Tensor:
        nfactors = len(factors)
        if nfactors == 2:
            return factors[0] @ factors[1]
        elif nfactors == 3:
            U, S, Vh = factors
            return (U * S) @ Vh
        else:
            raise ValueError('Unknown type of decomposition!')

        
class RandomizedMethod:
    _random_gens = RANDOM_GENS


class BaseSketch(Decomposer):    
    def __init__(self, sketch_size: Union[int, float] = None, 
                 compression_ratio: float = 0.5,
                 random_init: str = 'normal',
                 distortion_factor: float = 0.6):
        Decomposer.__init__(self, rank=None, distortion_factor=distortion_factor, random_init=random_init)
        
        assert 0 < compression_ratio < 1, 'compression_ratio must be in (0, 1)'
        self.compression_ratio = compression_ratio
        self.sketch_size = sketch_size
        self.original_matrix = None
    
    def evaluate_parameter(self, W: torch.Tensor, param_type: str) -> int:
        m, n = W.size(-2), W.size(-1)
        min_dim = min(m, n)
        
        if param_type == 'sketch_size':
            if self.sketch_size is not None:
                if isinstance(self.sketch_size, int):
                    return min(self.sketch_size, min_dim)
                else:
                    return max(1, int(self.sketch_size * min_dim))
            return max(1, int(min_dim * self.compression_ratio))
        else:
            return super().evaluate_parameter(W, param_type)
        
    def decompose(self, W: torch.Tensor, rank=None, *args, **kwargs):
        self.original_matrix = W.clone().detach()
        if rank is None:
            rank = self.evaluate_parameter(W, 'sketch_size', method='normal')
        elif isinstance(rank, float):
            rank = max(1, int(rank * min(W.size())))
        
        C = self.sketch(W, sketch_size=rank, *args, **kwargs)
        X = torch.linalg.lstsq(C, W).solution
        
        result = (C, X)
        return result


    def __call__(self, matrix: torch.Tensor, sketch_size: int = None, *args, **kwargs) -> torch.Tensor:
        self.original_matrix = matrix.clone().detach()
        return self.sketch(matrix, sketch_size, *args, **kwargs)

    @abstractmethod
    def _sketch(self, matrix: torch.Tensor, sketch_size: int, *args, **kwargs) -> torch.Tensor:
        pass
    
    def _decompose(self, W, rank, *args, **kwargs):
        return self.decompose(W, rank, *args, **kwargs)
    
    def _decompose_big(self, W, rank, *args, **kwargs):
        return self._decompose(W, rank, *args, **kwargs)

    def get_approximation_error(self, matrix: torch.Tensor, *decomposition_result) -> float:
        try:
            target_matrix = matrix if matrix is not None else self.original_matrix
            
            if target_matrix is None:
                raise ValueError("No matrix provided and no original matrix stored")
            
            if len(decomposition_result) == 1 and decomposition_result[0].shape == target_matrix.shape:
                reconstructed = decomposition_result[0]
            else:
                reconstructed = self.compose(*decomposition_result)
            
            return torch.linalg.norm(target_matrix - reconstructed, ord='fro').item()
        except:
            return float('inf')

    def compose(self, *factors, **kwargs) -> torch.Tensor:
        if len(factors) == 2:
            return factors[0] @ factors[1]
        elif len(factors) == 1:
            return factors[0]
        else:
            raise ValueError('Unknown sketch composition format')