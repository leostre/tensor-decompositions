from functools import partial
from typing import Callable, Optional

from torch.utils.checkpoint import checkpoint

import tdecomp
from tdecomp.grad_proj.tensorgrad.config import Galore2DProjectionSide
from tdecomp.grad_proj.tensorgrad.projectors.update_gap_scheduler import UpdateGapScheduler
from tdecomp.types import Number, TensorLike
import tensorly as tl

class GaLoreProjector:
    def __init__(self, 
                 rank: Number, 
                 verbose=False, 
                 svd_type: Optional[Callable[[TensorLike], tuple[TensorLike, TensorLike, TensorLike]]]=None, 
                 update_gap_scheduler: UpdateGapScheduler = UpdateGapScheduler(100, 1000), 
                 scale=1.0, 
                 galore_2d_proj_type: Galore2DProjectionSide = 'left', 
                 activation_checkpoint=False, 
                 support_complex=False
                 ):
        self.rank = rank
        self.verbose = verbose
        self.update_gap_scheduler = update_gap_scheduler
        self.scale = scale
        '''Scale used in back projections of tensor (P @ W * scale)'''
        self.ortho_matrix: TensorLike | tuple[TensorLike, TensorLike] = None
        self.galore_2d_proj_type: Galore2DProjectionSide = galore_2d_proj_type
        self.activation_checkpointing = activation_checkpoint
        '''Whether to use 'activation checkpointing' that reduce memory by creating callback function with tensor operands instead of immediate computation.
        https://docs.pytorch.org/docs/stable/checkpoint.html
        '''
        self.support_complex = support_complex
        '''Support of complex numbers'''
        self._reconstruction_buffer: TensorLike = None  # Pre-allocated reconstruction buffer
        self.svd_type = svd_type
        '''Kind of svd algorithm like randomized_svd or truncated_svd. Callable'''
        if verbose:
            print(f"rank={self.rank}, scale={self.scale}, galore_2d_proj_type={self.galore_2d_proj_type}, activation_checkpointing={self.activation_checkpointing}, support_complex={self.support_complex}")
            print(f"GaLoreProjector initialized with rank={self.rank}, scale={self.scale}, galore_2d_proj_type={self.galore_2d_proj_type}, activation_checkpointing={self.activation_checkpointing}, support_complex={self.support_complex}")


    def _project_right(self, full_rank_grad: TensorLike) -> TensorLike:
        low_rank_grad = optional_checkpoint_matmul(full_rank_grad, tl.transpose(self.ortho_matrix), self.activation_checkpointing)
        return low_rank_grad

    def _project_left(self, full_rank_grad: TensorLike) ->  TensorLike:
        low_rank_grad = optional_checkpoint_matmul(tl.transpose(self.ortho_matrix), full_rank_grad, self.activation_checkpointing)
        return low_rank_grad

    def _project_full(self, full_rank_grad: TensorLike) -> TensorLike:
        a = optional_checkpoint_matmul(tl.transpose(self.ortho_matrix[0]), full_rank_grad, self.activation_checkpointing)
        low_rank_grad = optional_checkpoint_matmul(a, tl.transpose(self.ortho_matrix[1]), self.activation_checkpointing)
        return low_rank_grad

    @tdecomp.utils.no_grad
    def project(self, full_rank_grad: TensorLike, iter: int) -> TensorLike:
        '''Main method for projecting gradients during model training'''
        type_ = self.galore_2d_proj_type
        if self.ortho_matrix is None or self.update_gap_scheduler.should_update(iter):
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, 
                                                           galore2dProjectionSide=type_) 
        low_rank_grad = getattr(self, f'_project_{type_}')(full_rank_grad)                              
        return low_rank_grad
    
    def _check_reconstruction_buffer_not_none(self, tensor: TensorLike):
        if self._reconstruction_buffer is None:
            self._reconstruction_buffer = tl.zeros(tl.shape(tensor), **tl.context(tensor))

    def _project_back_right(self, low_rank_grad: TensorLike) -> TensorLike:
        self._check_reconstruction_buffer_not_none(low_rank_grad)

        tl.matmul(low_rank_grad, self.ortho_matrix, out=self._reconstruction_buffer)
        return self._reconstruction_buffer * self.scale

    def _project_back_left(self, low_rank_grad: TensorLike) -> TensorLike:
        self._check_reconstruction_buffer_not_none(low_rank_grad)

        tl.matmul(self.ortho_matrix, low_rank_grad, out=self._reconstruction_buffer)
        return self._reconstruction_buffer * self.scale
    

    def _project_back_full(self, low_rank_grad: TensorLike) -> TensorLike:
        if self._reconstruction_buffer is None:
            self._reconstruction_buffer = tl.zeros((tl.shape(self.ortho_matrix[0])[0], tl.shape(self.ortho_matrix[1])[1]), 
                                                   **tl.context(low_rank_grad))
        
        intermediate = tl.matmul(self.ortho_matrix[0], low_rank_grad)
        tl.matmul(intermediate, self.ortho_matrix[1], out=self._reconstruction_buffer)
        return self._reconstruction_buffer * self.scale

    @tdecomp.utils.no_grad
    def project_back(self, low_rank_grad: TensorLike) -> TensorLike:
        return getattr(self, f'_project_back_{self.galore_2d_proj_type}')(low_rank_grad)
    
    @tdecomp.utils.no_grad
    def get_orthogonal_matrix(self, tensor: TensorLike, rank: Number, galore2dProjectionSide: Galore2DProjectionSide) -> TensorLike | tuple[TensorLike, TensorLike]:
        '''Returns ranked orthogonal matrix from SVD decomposition of `tensor`. If galore2dProjectionSide is `full` returns both U and Vh matricies, otherwise returns one.'''
        module_params = tensor
        original_context = tl.context(module_params)
        if tdecomp.utils.is_complex(module_params) and self.support_complex:
            float_data = False
            matrix = module_params + 0j
        elif not tdecomp.utils.is_floating_point(module_params):
            float_data = False
            matrix = module_params * 1.0, 
        else:
            float_data = True
            matrix = module_params

        full_n_params = tl.shape(matrix)[0] * tl.shape(matrix)[1]
        if isinstance(rank, float):
            low_rank_params = int(rank * full_n_params)
            int_rank = int(low_rank_params / tl.shape(matrix)[0])
        else:
            int_rank = rank

        if (self.svd_type is None):
            self.svd_type = partial(tl.truncated_svd, n_eigenvecs=min(tl.shape(tensor)))

            #make the smaller matrix always to be orthogonal matrix
        if galore2dProjectionSide == 'right':
            _, _, Vh = self.svd_type(matrix)
            B = Vh[:int_rank, :]
            if not float_data:
                B = tl.tensor(B, **original_context)
            return B
        elif galore2dProjectionSide == 'left':
            U, _, _ = self.svd_type(matrix)
            A = U[:, :int_rank]
            if not float_data:
                A = tl.tensor(A, **original_context)
            return A
        elif galore2dProjectionSide == 'full':
            U, _, Vh = self.svd_type(matrix)
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = tl.tensor(A, **original_context)
                B = tl.tensor(B, **original_context)
            return [A, B]
        else:
            raise ValueError('galore2dProjectionSide should be left, right or full')

def optional_checkpoint_matmul(a: TensorLike, b: TensorLike, activation_checkpoint=True) -> TensorLike:
    """optional_checkpoint_matmul performs torch.matmul and optionally performs
    activation checkpointing.
    """
    if activation_checkpoint:
        return checkpoint(tl.matmul, a, b) #TORCH dependent part of code
    else:
        return tl.matmul(a, b)
