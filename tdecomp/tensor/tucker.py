from typing import *

import tensorly as tl

import tdecomp
from tdecomp.types import SVDCallable, TensorDecompositionInit, TensorLike, Number
from tdecomp._base import TensorDecomposer
from tdecomp.matrix.decomposer import RandomizedSVD
from tdecomp.matrix.random_projections import Projector, ProjectorGenerator

__all__ = [
    'RPHOSVDDecomposition',
    'RSTHOSVDDecomposition',
    'RSTDecomposition'
]

class RPHOSVDDecomposition(TensorDecomposer):
    """
    Random Projection Higher Order Singular Value Decomposition (RP-HOSVD)
    
    This algorithm performs HOSVD decomposition using random projections for efficiency.
    The algorithm works by:
    1. For each tensor mode:
       - Transpose tensor to put current mode first
       - Apply random projection
       - Perform QR decomposition
    2. Compute core tensor through tensor contractions
    
    Args:
        rank: target rank for each mode (can be list or int)
        distortion_factor: distortion factor for random projection
        power: power iteration parameter for random projection
        random_init: type of random initialization ('normal' or 'ortho')
    """
    
    def __init__(self, *, rank: Optional[Union[Number, List[Number]]] = None, 
                 distortion_factor: float = 0.6,
                 power: int = 3,
                 random_init: ProjectorGenerator = ProjectorGenerator.normal):
        super().__init__(rank=rank, random_init=random_init)
        self.power = power
        self.projector = Projector(random_init)

    def _decompose(self, X: TensorLike, rank: List[int], **kwargs) -> tuple[TensorLike, list[TensorLike]]:
        
        factor_matrices = []

        for mode_idx in range(tl.ndim(X)):
            unfold_tensor = tl.unfold(X, mode_idx)
            projected_matrix = self.projector.rproject(unfold_tensor, rank[mode_idx], renew=True)

            # Perform QR decomposition on the projection matrix
            Q, _ = tl.qr(tl.transpose(projected_matrix), mode='reduced')  # QR on transposed projection
            #from functools import reduce
            # assert tuple(Q.shape) == (reduce(int.__mul__, self.shape) // tensor.shape[mode_idx], rank[mode_idx])

            X = tl.tenalg.mode_dot(X, tl.transpose(Q), mode_idx)
            
            # Store factor matrix (Q has shape (mode_size, target_rank))
            factor_matrices.append(Q)
            
        
        # The final current_tensor is the core tensor
        
        return X, factor_matrices


class RSTHOSVDDecomposition(TensorDecomposer):
    """
    Randomized Sequentially Truncated HOSVD (R-STHOSVD) Algorithm
    
    This algorithm performs HOSVD decomposition using randomized SVD for each mode.
    The algorithm works by:
    1. For each tensor mode:
       - Apply Basic Randomized SVD to the n-unfolding matrix
       - Update the core tensor through tensor contraction
    2. Return the core tensor and factor matrices
    
    Args:
        rank: target rank for each mode (can be list or int)
        oversampling: oversampling parameter for randomized SVD
        power_iteration: power iteration parameter for randomized SVD
        random_init: type of random initialization ('normal' or 'ortho')
    """
    
    def __init__(self, *, rank: Optional[Union[Number, List[Number]]] = None,
                 oversampling: int = 10,
                 power_iteration: int = 2,
                 distortion_factor: float = 0.1,
                 random_init: ProjectorGenerator = ProjectorGenerator.normal):
        super().__init__(rank=rank, random_init=random_init)
        self.oversampling = oversampling
        self.power_iteration = power_iteration
        self.rsvd = RandomizedSVD(
            power=self.power_iteration,
            distortion_factor=distortion_factor,
            random_init=random_init
        )
    
    def _decompose(self, X: TensorLike, rank: List[int], **kwargs) -> tuple[TensorLike, list[TensorLike]]:
        """
        Decompose tensor using R-STHOSVD
        
        Args:
            tensor: input tensor to decompose
            
        Returns:
            tuple: (core_tensor, factor_matrices)
        """

        # Initialize core tensor and factor matrices
        core_tensor = tl.tensor(X, **tl.context(X))
        self.original_shape = tl.shape(X)
        factor_matrices = []
        
        # Process each mode in reverse order to get correct core tensor shape
        for mode_idx in range(tl.ndim(X)):
            ort = tl.unfold(X, mode_idx)
            U, *_= self.rsvd.decompose(ort, rank[mode_idx])
            factor_matrices.append(U)
            core_tensor = tl.tenalg.mode_dot(core_tensor, tl.transpose(U), mode_idx)
        
        return core_tensor, factor_matrices


class RSTDecomposition(TensorDecomposer):
    """
    Randomized Sampling Tucker Approximation (R-ST) Algorithm
    
    This algorithm performs Tucker decomposition using random sampling of columns
    from tensor unfoldings. The algorithm works by:
    1. For each mode n = 1, 2, ..., N:
       - Sample columns from X(n) based on probability distribution
       - Store them in factor matrix Q(n) ∈ ℝ^(In × Rn)
    2. Compute core tensor S = X ×₁ Q₁^† ×₂ Q₂^† ... ×ₙ Qₙ^†
    
    Args:
        rank: target rank for each mode (can be list or int)
        sampling_method: method for column sampling ('uniform', 'norm_based', 'leverage_score')
        distortion_factor: distortion factor for random projection
        random_init: type of random initialization ('normal' or 'ortho')
    """
    
    def __init__(self, *, rank: Optional[Union[Number, List[Number]]] = None,
                 sampling_method: str = 'norm_based',
                 distortion_factor: float = 0.6,
                 random_init: ProjectorGenerator = ProjectorGenerator.normal):
        super().__init__(random_init=random_init, rank=rank)
        self.rsvd = RandomizedSVD(
            None, distortion_factor=distortion_factor, random_init=random_init
        )

    def _sample(self, tensor: TensorLike, n: int) -> TensorLike:
        return tensor[..., tdecomp.utils.randperm(tensor.shape[-1], tl.context(tensor))[:n]]
        
    
    def _decompose(self, X: TensorLike, rank: List[int], **kwargs) -> tuple[TensorLike, list[TensorLike]]:
        """
        Decompose tensor using R-ST algorithm
        
        Args:
            tensor: input tensor to decompose
            
        Returns:
            tuple: (core_tensor, factor_matrices)
        """
        
        core_tensor = tl.tensor(X, **tl.context(X))
        factor_matrices = []
        
        # Step 1: For each mode n = 1, 2, ..., N
        for mode_idx in range(tl.ndim(X)):
            ort = tl.unfold(X, mode_idx)
            Q = self._sample(ort, n=rank[mode_idx])
            factor_matrices.append(Q)
            U, S, Vh = self.rsvd.decompose(Q)
            Q_inv = self.rsvd.compose(tl.transpose(Vh), 1 / S, tl.transpose(U))
            core_tensor = tl.tenalg.mode_dot(core_tensor, Q_inv, mode_idx)
        
        return core_tensor, factor_matrices


class HOOIDecomposition(TensorDecomposer):
    '''https://arxiv.org/abs/2110.12564
    
    References
    ----------
    .. [1] tl.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    '''
    def __init__(self, 
                 rank: Optional[Number | List[Number]] = None,
                 init: TensorDecompositionInit = 'svd',
                 n_iter_max: int = 100,
                 svd_type: tl.tenalg.svd.SVD_TYPES | SVDCallable = 'truncated_svd',
                 ):
        super().__init__(rank)
        self.init = init
        '''How to init core and factors: with default Tucker via SVD and U, via pure random tensors or via passed objects'''
        self.n_iter_max = n_iter_max
        '''Iterations for convergens of algorithm'''
        self.svd_type = svd_type
        '''SVD used in calculation of ranked U (factors) tensors each iteration in each mode'''

    def decompose(self, 
                  tensor: TensorLike, 
                  rank: Optional[Number | List[Number]] = None,
                  init: Optional[TensorDecompositionInit] = None,
                  n_iter_max: Optional[int] = None,
                  svd_type: Optional[tl.tenalg.svd.SVD_TYPES | SVDCallable] = None,
                  **kwargs
                  ) -> tuple[TensorLike, list[TensorLike]]:
        init = init if init is not None else self.init # type: ignore
        n_iter_max = n_iter_max if n_iter_max is not None else self.n_iter_max
        svd_type = svd_type if svd_type is not None else self.svd_type # type: ignore
        return super().decompose(tensor, rank, init=init, n_itermax=n_iter_max, svd_type=svd_type)

    def _decompose(self, X: TensorLike, rank: List[int], **kwargs) -> tuple[TensorLike, list[TensorLike]]:
        core, factors = tl.decomposition.tucker(X, rank=rank, **kwargs)
        return core, factors

__local_names = locals()

DECOMPOSERS: Dict[str, type[TensorDecomposer]]= {
    name: __local_names[name] for name in __all__
}
