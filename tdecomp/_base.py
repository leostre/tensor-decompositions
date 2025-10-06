import torch

from functools import wraps, reduce, partial, partialmethod
from typing import *
from abc import ABC, abstractmethod

from tensorly import set_backend
from tensorly.tenalg import mode_dot

from tdecomp.matrix.random_projections import RANDOM_GENS

set_backend('pytorch')

Number = Union[int, float]

DIM_SUM_LIM = 1024
DIM_LIM = 1024

def _need_t(f):
    """Performs matrix transposition for maximal projection effect"""
    @wraps(f)
    def _wrapper(self: Decomposer, W: torch.Tensor, *args, **kwargs):
            m, n = W.size(-2), W.size(-1)
            _is_transposed = m >= n
            weight = W.t() if _is_transposed else W
            tns = f(self, weight, *args, **kwargs)
            return (
                tns if not _is_transposed
                else tuple(t.t() for t in reversed(tns))
            )
    return _wrapper

def _conditioning(f):
    """If conditioner is detected, apply W' = W @ C @ C^-1
    then U, S, Vh = Decompostion(W @ C)
    and Vh = Vh @ C^-1
    """
    @wraps(f)
    def _conditioned(self: "Decomposer", W: torch.Tensor, rank=None, conditioner=None, *args, **kwargs):
        if conditioner is None:
            conditioner = self._conditioner
        if conditioner is None:
            return f(self, W, rank, *args, **kwargs)
        if conditioner.ndim != 1:
            operation = torch.matmul
            inverse_operation = torch.linalg.pinv
        else: 
            operation = torch.mul
            inverse_operation = lambda x: 1 / x  
        W = operation(W, conditioner)
        inverse = inverse_operation(conditioner)
        *decomposition, Vh = f(self, W, rank, *args, **kwargs)
        Vh = operation(Vh, inverse)
        return *decomposition, Vh
    return _conditioned

class Decomposer(ABC):
    def __init__(self, rank: Union[int, float] = None, distortion_factor: float = 0.6, 
                 random_init: str = 'normal'):
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.distortion_factor = distortion_factor
        self.random_init = random_init
        self.rank = rank
        self._conditioner = None

    def _get_rank(self, tensor: torch.Tensor, rank: Number) -> int:
        rank = rank or self.rank
        if rank is None:
            rank = self.estimate_stable_rank(tensor)
        elif isinstance(rank, float):
            rank = max(1, int(rank * min(tensor.size())))
        elif isinstance(rank, int):
            rank = min(rank, min(tensor.size()))
        else:
            raise TypeError(f'Expected types for `rank`: {repr(Number)}, got `{type(rank)}`')
        return rank                

    @_conditioning
    def decompose(self, tensor: torch.Tensor, rank: Number = None, *args, **kwargs):
        rank = self._get_rank(tensor, rank)
        if not self._is_big(tensor):
            return self._decompose(tensor, rank, *args, **kwargs)
        else:
            return self._decompose_big(tensor, rank, *args, **kwargs)
        
    def _is_big(self, W: torch.Tensor):
        return sum(W.size()) > DIM_SUM_LIM or any(d > DIM_LIM for d in W.size())
    
    def set_conditioner(self, conditioner):
        self._conditioner = conditioner
        
    @abstractmethod
    def _decompose(self, W, rank, *args, **kwargs):
        pass
    
    def _decompose_big(self, W, rank, *args, **kwargs):
        return self._decompose(W, rank, *args, **kwargs)
    
    def estimate_stable_rank(self, W):
        n_samples = max(W.shape)
        eps = self.distortion_factor
        min_num_samples = torch.ceil(4 * torch.log(torch.scalar_tensor(n_samples)) / (eps**2 / 2 - eps**3 / 3))
        return max(min(torch.round(min_num_samples), *W.size()), 1)
    
    def get_approximation_error(self, tensor: torch.Tensor, *approximation, relative: bool = True):
        eps = 1e-5
        approximation = self.compose(*approximation)
        error_mtr = tensor - approximation
        error_norm = torch.linalg.norm(error_mtr)
        if relative:
            initial_norm = torch.linalg.norm(tensor)
            error_norm /= initial_norm + eps
        return error_norm
    
    def compose(self, *factors, **kwargs) -> torch.Tensor:
        nfactors = len(factors)
        if nfactors == 2:
            return factors[0] @ factors[1]
        elif nfactors == 3:
            U, S, Vh = factors
            return (U * S) @ Vh
        else:
            raise ValueError('Unknown type of decomposition!')


class TensorDecomposer(Decomposer):
    def _get_rank(self, tensor: torch.Tensor, rank: Number) -> List[int]:
        rank = rank or self.rank
        if rank is None:
            rank = list(tensor.size())
        elif isinstance(rank, int):
            rank = [rank] * tensor.ndim
        elif isinstance(rank, float):
            assert 0 < rank <= 1, 'Float rank must lie in (0, 1]'
            rank = int(rank * min(tensor.size()))
            rank = [rank] * tensor.ndim
        elif hasattr(rank, '__iter__'):
            if len(rank) != tensor.ndim:
                raise ValueError(f"Rank list length {len(rank)} must match tensor dimensions {tensor.dim()}")
            ranks = [None] * tensor.ndim
            for i in range(tensor.ndim):
                if isinstance(rank[i], int):
                    ranks[i] = rank[i]
                elif isinstance(rank[i], float):
                    ranks[i] = int(rank[i] * tensor.size(i))
                else:
                    raise ValueError('Unexpected value for rank!')
            rank = ranks
        else:
            raise TypeError(f'Supprted formats are: int, float (0,1] and lists of them, got {type(rank)}')
        return rank

    def compose(self, core: torch.Tensor, *factors: List[torch.Tensor]) -> torch.Tensor:
        for i, factor in enumerate(factors):
            core = mode_dot(core, factor, i)
        return core
    
    def get_approximation_error(self, tensor, *approximation, relative = True):
        core, factors = approximation
        return super().get_approximation_error(tensor, core, *factors, relative=relative)
