from typing import *

from tensorly import set_backend
from tensorly.tenalg import mode_dot
from tensorly.base import unfold
import torch

from tdecomp._base import TensorDecomposer, Number
from tdecomp.matrix.decomposer import RandomizedSVD
from tdecomp.matrix.random_projections import Projector

set_backend('pytorch')


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
    
    def __init__(self, *, rank: Optional[Union[int, List[int]]] = None, 
                 distortion_factor: float = 0.6,
                 power: int = 3,
                 random_init: str = 'normal'):
        super().__init__(rank=rank, distortion_factor=distortion_factor, random_init=random_init)
        self.power = power
        self.projector = Projector(random_init)

    def _decompose(self, tensor: torch.Tensor, rank: List[int]) -> tuple:
        
        self.shape = list(tensor.shape)
        factor_matrices = []

        for mode_idx in range(tensor.dim()):
            unfold_tensor = unfold(tensor, mode_idx)
            projected_matrix = self.projector.rproject(unfold_tensor, rank[mode_idx], renew=True)

            # Perform QR decomposition on the projection matrix
            Q, R = torch.linalg.qr(projected_matrix.T, mode='reduced')  # QR on transposed projection
            from functools import reduce
            # assert tuple(Q.shape) == (reduce(int.__mul__, self.shape) // tensor.shape[mode_idx], rank[mode_idx])

            tensor = mode_dot(tensor, Q.T, mode_idx)
            
            # Store factor matrix (Q has shape (mode_size, target_rank))
            factor_matrices.append(Q)
            
            # Calculate new shape for the contracted tensor
            self.shape[mode_idx] = rank[mode_idx]
        
        # The final current_tensor is the core tensor
        
        return tensor, factor_matrices


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
    
    def __init__(self, *, rank: Optional[Union[int, List[int]]] = None,
                 oversampling: int = 10,
                 power_iteration: int = 2,
                 distortion_factor: float = 0.1,
                 random_init: str = 'normal'):
        super().__init__(rank=rank, distortion_factor=distortion_factor, random_init=random_init)
        self.oversampling = oversampling
        self.power_iteration = power_iteration
        self.rsvd = RandomizedSVD(
            power=self.power_iteration,
            distortion_factor=distortion_factor,
            random_init=random_init
        )
    
    def _decompose(self, tensor: torch.Tensor, rank: List[int]) -> tuple:
        """
        Decompose tensor using R-STHOSVD
        
        Args:
            tensor: input tensor to decompose
            
        Returns:
            tuple: (core_tensor, factor_matrices)
        """

        # Initialize core tensor and factor matrices
        core_tensor = tensor.clone()
        self.original_shape = tensor.shape
        factor_matrices = []
        
        # Process each mode in reverse order to get correct core tensor shape
        for mode_idx in range(tensor.dim()):
            ort = unfold(tensor, mode_idx)
            U, *_= self.rsvd.decompose(ort)
            factor_matrices.append(U)
            core_tensor = mode_dot(core_tensor, U, mode_idx)
        
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
    
    def __init__(self, *, rank: Optional[Union[int, List[int]]] = None,
                 sampling_method: str = 'norm_based',
                 distortion_factor: float = 0.6,
                 random_init: str = 'normal'):
        super().__init__(random_init=random_init, rank=rank, distortion_factor=distortion_factor)
    
    def _decompose(self, tensor: torch.Tensor, rank: List[int]) -> tuple:
        """
        Decompose tensor using R-ST algorithm
        
        Args:
            tensor: input tensor to decompose
            
        Returns:
            tuple: (core_tensor, factor_matrices)
        """
        
        # Store original tensor shape
        self.original_shape = tensor.shape
        core_tensor = tensor
        factor_matrices = []
        
        # Step 1: For each mode n = 1, 2, ..., N
        for mode_idx in range(tensor.dim()):
            ort = unfold(tensor, mode_idx)
            indices = torch.randperm(ort.shape[-1])
            new_tensor = ort[:, indices].T
            factor_matrices.append(new_tensor)
            new_tensor = torch.linalg.pinv(new_tensor)
            core_tensor = mode_dot(core_tensor, new_tensor, mode_idx)
        
        return core_tensor, factor_matrices
    


__local_names = locals()

DECOMPOSERS: Dict[str, TensorDecomposer]= {
    name: __local_names[name] for name in __all__
}
