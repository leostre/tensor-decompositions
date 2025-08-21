import torch

from functools import wraps, reduce, partial
from typing import *
from abc import ABC, abstractmethod


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
            rank = self.estimate_stable_rank(W)
        elif isinstance(rank, float):
            rank = max(1, int(rank * min(W.size())))
        if not self._is_big(W):
            return self._decompose(W, rank, *args, **kwargs)
        else:
            return self._decompose_big(W, rank, *args, **kwargs)
        
    def _is_big(self, W: torch.Tensor):
        return sum(W.size()) > DIM_SUM_LIM or any(d > DIM_LIM for d in W.size())

    @abstractmethod
    def _decompose(self, W, rank, *args, **kwargs):
        pass
    
    def _decompose_big(self, W, rank, *args, **kwargs):
        return self._decompose(W, rank, *args, **kwargs)
    
    def estimate_stable_rank(self, W):
        n_samples = max(W.shape)
        eps = self.distortion_factor
        min_num_samples = torch.ceil(4 * torch.log(n_samples) / (eps**2 / 2 - eps**3 / 3))
        return min((round(max(min_num_samples)), *W.size(), 1))
    
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
        
class BaseSketch(Decomposer):    
    def __init__(self, sketch_size: Union[int, float] = None, 
                 compression_ratio: float = 0.5,
                 random_init: str = 'normal',
                 distortion_factor: float = 0.6):
        Decomposer.__init__(self, rank=None, distortion_factor=distortion_factor, random_init=random_init)
        
        assert 0 < compression_ratio < 1, 'compression_ratio must be in (0, 1)'
        self.compression_ratio = compression_ratio
        self.sketch_size = sketch_size

    def sketch(self, matrix: torch.Tensor, sketch_size: int = None, *args, **kwargs) -> torch.Tensor:
        if sketch_size is None:
            sketch_size = self.estimate_sketch_size(matrix)
        elif isinstance(sketch_size, float):
            sketch_size = max(1, int(sketch_size * min(matrix.size())))
            
        if not self._is_big(matrix):
            return self._sketch(matrix, sketch_size, *args, **kwargs)
        else:
            return self._sketch_big(matrix, sketch_size, *args, **kwargs)

    def __call__(self, matrix: torch.Tensor, sketch_size: int = None, *args, **kwargs) -> torch.Tensor:
        return self.sketch(matrix, sketch_size, *args, **kwargs)

    def decompose(self, W: torch.Tensor, rank=None, *args, **kwargs):
        sketch_result = self.sketch(W, sketch_size=rank, *args, **kwargs)
        
        if hasattr(self, '_as_decomposition'):
            return self._as_decomposition(sketch_result, W)
        else:
            I = torch.eye(sketch_result.size(1), device=W.device, dtype=W.dtype)
            return sketch_result, I

    @abstractmethod
    def _sketch(self, matrix: torch.Tensor, sketch_size: int, *args, **kwargs) -> torch.Tensor:
        pass
    
    def _sketch_big(self, matrix: torch.Tensor, sketch_size: int, *args, **kwargs) -> torch.Tensor:
        return self._sketch(matrix, sketch_size, *args, **kwargs)

    def _decompose(self, W, rank, *args, **kwargs):
        return self.decompose(W, rank, *args, **kwargs)
    
    def _decompose_big(self, W, rank, *args, **kwargs):
        return self._decompose(W, rank, *args, **kwargs)

    def estimate_sketch_size(self, matrix: torch.Tensor) -> int:
        n, m = matrix.shape
        min_dim = min(n, m)
        
        if self.sketch_size is not None:
            if isinstance(self.sketch_size, int):
                return min(self.sketch_size, min_dim)
            else:
                return max(1, int(self.sketch_size * min_dim))
        
        eps = self.compression_ratio
        min_sketch_size = torch.ceil(4 * torch.log(torch.tensor(min_dim)) / (eps**2 / 2 - eps**3 / 3))
        return min(int(min_sketch_size), min_dim, 1000)

    def get_approximation_error(self, matrix: torch.Tensor, sketch: torch.Tensor) -> float:
        if sketch.shape == matrix.shape:
            return torch.linalg.norm(matrix - sketch).item()
        
        try:
            reconstructed = self.compose(*self.decompose(matrix))
            return torch.linalg.norm(matrix - reconstructed).item()
        except:
            return float('inf')

    def compose(self, *factors, **kwargs) -> torch.Tensor:
        if len(factors) == 2:
            return factors[0] @ factors[1]
        elif len(factors) == 1:
            return factors[0]
        else:
            raise ValueError('Unknown sketch composition format')