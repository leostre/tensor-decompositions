import math
from typing import Any, List, Optional, Union

import tensorly as tl

import tdecomp
tl.set_backend('pytorch') #TODO think about place of it
type TensorLike = Any
'''Tensorly supports work with different tensor backends (numpy, torch.tensor and so on), 
but it doesnt describe abstract class for it. 
So the tensor can be of `Any` type depending on backend setted in `tl.set_backend` .'''

from functools import wraps
from abc import ABC, abstractmethod
from tdecomp.matrix.random_projections import ProjectorGenerator
from tdecomp.matrix.importance_generators import ColumnRowImportancesGenerator

__all__ = [
    'Number',
    'Decomposer',
    'TensorDecomposer'
]

Number = Union[int, float]

DIM_SUM_LIM = 1024
DIM_LIM = 1024

def _need_t(f):
    """Performs matrix transposition for maximal projection effect  
    Supports only 2D tensors!"""
    @wraps(f)
    def _wrapper(self: Decomposer, W: TensorLike, *args, **kwargs) -> tuple[TensorLike, TensorLike, TensorLike]:
        m, n = tl.shape(W)[-2], tl.shape(W)[-1]
        _is_transposed = m >= n
        weight = tl.transpose(W) if _is_transposed else W
        decomposed_tensors = f(self, weight, *args, **kwargs)
        return (
            decomposed_tensors if not _is_transposed
            else tuple(tl.transpose(t) for t in reversed(decomposed_tensors))
        )
    return _wrapper

def _conditioning(f):
    """If conditioner is detected, apply W' = W @ C @ C^-1
    then U, S, Vh = Decompostion(W @ C)
    and Vh = Vh @ C^-1  
    Conditioner shouldn't be singular or contain zero columns!  
    C could be 1D or 2D
    """
    @wraps(f)
    def _conditioned(self: "Decomposer", W: TensorLike, rank=None, conditioner=None, *args, **kwargs) -> tuple[TensorLike, TensorLike, TensorLike]:
        if conditioner is None:
            conditioner = self._conditioner
        if conditioner is None:
            return f(self, W, rank, *args, **kwargs)
        if tl.ndim(conditioner) != 1:
            W = tl.matmul(W, conditioner)
            inverse_conditioner = tdecomp.utils.pseudo_inverse(conditioner)
            *decomposition, Vh = f(self, W, rank, *args, **kwargs)
            Vh = tl.matmul(Vh, inverse_conditioner)
            return *decomposition, Vh
        else: 
            W = tl.einsum('ij,j->ij', W, conditioner)
            inverse = 1 / conditioner #WARN: also can be failed (devide by zero)
            *decomposition, Vh = f(self, W, rank, *args, **kwargs)
            Vh = tl.einsum('ij,j->ij', Vh, inverse)
            return *decomposition, Vh
    return _conditioned

class Decomposer(ABC):
    def __init__(self, rank: Optional[Number] = None, distortion_factor: float = 0.6, 
                 random_init: (ProjectorGenerator | ColumnRowImportancesGenerator) = ProjectorGenerator.normal):
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.distortion_factor = distortion_factor
        self.random_init = random_init
        self.rank = rank
        self._conditioner = None

    def _get_rank(self, tensor: TensorLike, rank: Optional[Number]) -> int:
        rank = rank or self.rank
        if rank is None:
            rank = self.estimate_stable_rank(tensor)
        elif isinstance(rank, float):
            rank = max(1, int(rank * min(tl.shape(tensor))))
        elif isinstance(rank, int):
            rank = min(rank, min(tl.shape(tensor)))
        else:
            raise TypeError(f'Expected types for `rank`: {repr(Number)}, got `{type(rank)}`')
        return rank

    @_conditioning
    def decompose(self, tensor: TensorLike, rank: Optional[Number] = None, *args, **kwargs) -> tuple[TensorLike, TensorLike, TensorLike]:
        rank = self._get_rank(tensor, rank)
        if not self._is_big(tensor):
            return self._decompose(tensor, rank, *args, **kwargs)
        else:
            return self._decompose_big(tensor, rank, *args, **kwargs)
        
    def _is_big(self, W: TensorLike):
        return sum(tl.shape(W)) > DIM_SUM_LIM or any(d > DIM_LIM for d in tl.shape(W))
    
    def set_conditioner(self, conditioner: TensorLike):
        self._conditioner = conditioner
        
    @abstractmethod
    def _decompose(self, W: TensorLike, rank: int, *args, **kwargs) -> tuple[TensorLike, TensorLike, TensorLike]:
        pass
    
    def _decompose_big(self, W: TensorLike, rank: int, *args, **kwargs) -> tuple[TensorLike, TensorLike, TensorLike]:
        return self._decompose(W, rank, *args, **kwargs)
    
    def estimate_stable_rank(self, W: TensorLike):
        n_samples = max(tl.shape(W))
        eps = self.distortion_factor
        min_num_samples = int(4 * math.log(n_samples) / (eps**2 / 2 - eps**3 / 3))
        return max(min(min_num_samples, *tl.shape(W)), 1)
    
    def get_approximation_error(self, tensor: TensorLike, *approximation_matrices, relative: bool = True) -> float:
        eps = 1e-5
        approximation = self.compose(*approximation_matrices)
        error_mtr = tensor - approximation
        error_norm = tl.norm(error_mtr, order=2)
        if relative:
            initial_norm = tl.norm(tensor, order=2)
            error_norm /= initial_norm + eps
        return error_norm
    

    def compose(self, *factors, **kwargs) -> TensorLike:
        '''
        :param factors: U S and Vh or US and Vh. Works only with 2D tensors, S treated as 1D diagonal.
        '''
        nfactors = len(factors)
        if nfactors == 2:
            return tl.matmul(factors[0], factors[1])
        elif nfactors == 3:
            U, S, Vh = factors
            US = tl.einsum("ij,j->ij", U, S)
            return tl.matmul(US, Vh)
        else:
            raise ValueError('Unknown type of decomposition!')


class TensorDecomposer(Decomposer):
    def _get_tensor_rank(self, tensor: TensorLike, rank: Optional[Union[Number, List]]) -> List[int]:
        '''Apply rank to tensor. Used word "rank" in term of shape, not a lineary-independent tensor basis (not a matrix rank).
        '''
        rank = rank or self.rank
        tensor_ndim = tl.ndim(tensor)
        if rank is None:
            rank = list(tl.shape(tensor))
        elif isinstance(rank, int):
            rank = [rank] * tensor_ndim
        elif isinstance(rank, float):
            assert 0 < rank <= 1, 'Float rank must lie in (0, 1]'
            rank = int(rank * min(tl.shape(tensor)))
            rank = [rank] * tensor_ndim
        elif hasattr(rank, '__iter__'):
            if len(rank) != tensor_ndim:
                raise ValueError(f"Rank list length {len(rank)} must match tensor dimensions {tensor_ndim}")
            ranks = [None] * tensor_ndim
            for i in range(tensor_ndim):
                if isinstance(rank[i], int):
                    ranks[i] = rank[i]
                elif isinstance(rank[i], float):
                    ranks[i] = int(rank[i] * tl.shape(tensor)[i])
                else:
                    raise ValueError('Unexpected value for rank!')
            rank = ranks
        else:
            raise TypeError(f'Supported formats are: int, float (0,1] and lists of them, got {type(rank)}')
        return rank # type: ignore

    def compose(self, core: TensorLike, *factors) -> TensorLike:
        for i, factor in enumerate(factors):
            core = tl.tenalg.mode_dot(core, factor, i)
        return core
    
    def get_approximation_error(self, tensor: TensorLike, *approximation, relative = True):
        core, factors = approximation
        return super().get_approximation_error(tensor, core, *factors, relative=relative)