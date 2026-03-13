from typing import Optional, Union, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import CP
from tensorly.cp_tensor import cp_to_tensor, CPTensor

from tdecomp._base import TensorDecomposer
from tdecomp.types import TensorLike
from tdecomp.matrix.random_projections import ProjectorGenerator

class CPDecomposition(TensorDecomposer):
    def __init__(self, 
                 rank: Optional[Union[int, float, List[int]]] = None,
                 random_init=ProjectorGenerator.normal,
                 n_iter_max: int = 100,
                 tol: float = 1e-6,
                 init: str = 'random',
                 normalize_factors: bool = False,
                 linesearch: bool = False):
        super().__init__(rank=rank, random_init=random_init)
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.init = init
        self.normalize_factors = normalize_factors
        self.linesearch = linesearch

    def decompose(self, X: TensorLike, rank: List[int], **kwargs):
        cp_rank = rank[0] if isinstance(rank, (list, tuple)) else rank
        
        n_iter_max = kwargs.get('n_iter_max', self.n_iter_max)
        tol = kwargs.get('tol', self.tol)
        init = kwargs.get('init', self.init)
        normalize_factors = kwargs.get('normalize_factors', self.normalize_factors)
        linesearch = kwargs.get('linesearch', self.linesearch)
        random_state = kwargs.get('random_state', None)

        original_device = X.device
        if original_device.type != 'cpu':
            X = X.cpu()
        
        cp_decomp = CP(
            rank=cp_rank,
            n_iter_max=n_iter_max,
            tol=tol,
            init=init,
            normalize_factors=normalize_factors,
            linesearch=linesearch,
            random_state=random_state,
        )
        
        cp_tensor: CPTensor = cp_decomp.fit_transform(X)
        
        weights = cp_tensor.weights
        factors = cp_tensor.factors
        
        self.weights_ = weights
        self.factors_ = factors
        self.cp_tensor_ = cp_tensor
        self.n_iterations_ = n_iter_max 
        
        return weights, factors
    
    def _decompose_big(self, X, rank, **kwargs):
        kwargs['n_iter_max'] = kwargs.get('n_iter_max', self.n_iter_max * 2)
        return self._decompose(X, rank, **kwargs)
    
    def compose(self, core, *factors):
        return cp_to_tensor((core, list(factors)))
    
    def _decompose(self, X, rank, **kwargs):
        return super()._decompose(X, rank, **kwargs)
    
    def get_approximation_error(self, tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, 
                                *factors: torch.Tensor, relative: bool = True):
        if weights is None:
            weights = self.weights_
            factors_list = self.factors_
        else:
            factors_list = list(factors)
        
        reconstructed = self.compose(weights, *factors_list)
        
        error = torch.norm(tensor - reconstructed)
        if relative:
            error = error / torch.norm(tensor)
        
        return error.item()
    