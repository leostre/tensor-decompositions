from typing import *

import tensorly as tl
from torch.autograd.profiler import record_function

import tdecomp
from tdecomp.grad_proj.tensorgrad.projectors.update_gap_scheduler import UpdateGapScheduler
from tdecomp.tensor.tucker import HOOIDecomposition
from tdecomp.types import TensorLike
import torch


class TensorGradLowRankProjector:
    def __init__(
        self, 
        rank, 
        update_gap_scheduler: UpdateGapScheduler,  
        verbose=False, 
        scale=1.0, 
        warm_restart=False,
        n_iter_max=10,
        svd_type: tl.tenalg.svd.SVD_TYPES | Callable[[TensorLike], tuple[TensorLike, TensorLike, TensorLike]] = "truncated_svd",
        tensor_decomposer_type: type[HOOIDecomposition] = HOOIDecomposition
    ):
        """
        Args:
            rank: Target rank.
            update_gap_scheduler: Instance of UpdateGapScheduler.
            verbose: If True, prints diagnostic messages.
            scale: Scaling factor applied after back projection.
            warm_restart: Continue from the previous projection when updating.
            n_iter_max: Maximum number of iterations for the Tucker decomposition.
            svd_type: Type of SVD to use for Tucker decomposition. Options: "randomized_svd" or "truncated_svd".
        """
        # Then initialize other attributes
        self.rank = rank
        self.verbose = verbose
        self.scale = scale
        self.proj_tensor = None
        self.warm_restart = warm_restart
        self.n_iter_max = n_iter_max
        self.update_gap_scheduler = update_gap_scheduler
        self.num_updates = 0
        self.num_steps = 0
        self._rank_validated = False
        self.svd_type = svd_type
        self.tensor_decomposer = tensor_decomposer_type(rank=self.rank, init='svd', n_iter_max=self.n_iter_max, svd_type=self.svd_type)
        if self.verbose:
            print(f"TensorGradLowRankProjector initialized with rank={self.rank}, scale={self.scale}, warm_restart={self.warm_restart}, n_iter_max={self.n_iter_max}, svd_type={self.svd_type}")
        
    def should_update_projector(self, iter: int) -> bool:
        return self.update_gap_scheduler.should_update(iter)

    @tdecomp.utils.no_grad
    def project(self, full_rank_grad: TensorLike, iter: int):
        with record_function("### TENSOR_GRAD_PROJECT_FORWARD"):
            if self.proj_tensor is None or self.should_update_projector(iter):
                self.proj_tensor = self.get_projection_tensor(full_rank_grad)
                self.num_updates += 1

            self.num_steps = iter
            return self.transform(self.proj_tensor, full_rank_grad)

    @tdecomp.utils.no_grad
    def project_back(self, low_rank_grad, output_buffer=None, alpha=1.0, accumulate=False):
        with record_function("#### TENSOR_GRAD_PROJECT_BACK"):
            # If out is provided, use it as the output buffer
            if output_buffer is not None:
                # Apply inverse transform with the provided buffer
                self.inverse_transform(self.proj_tensor, low_rank_grad, output_buffer=output_buffer, alpha=alpha*self.scale)
                return output_buffer
            else:
                # No buffer provided, let inverse_transform allocate a new tensor
                full_rank_grad = self.inverse_transform(self.proj_tensor, low_rank_grad)
                return full_rank_grad * self.scale
    
    
    # Tucker decomp: higher-order SVD
    def get_projection_tensor(self, weights: TensorLike) -> list[TensorLike]:
        matrix = weights
        original_dtype = tl.context(matrix)["dtype"]
                
        # Always use full precision for tucker decomposition
        if tdecomp.utils.is_complex(matrix):
            if tl.context(matrix)["dtype"] is not torch.complex64:
                matrix = tl.tensor(matrix, dtype=torch.complex64)
            
            
        # Handle initialization with warm restart
        if self.warm_restart and self.proj_tensor is not None:
            # Convert factors to full precision temporarily for initialization
            # check if on same device as matrix (upd: dont check on keras!)
            factors = self.proj_tensor
            # check if full precision if not convert to full precision - check if complex32
            if tdecomp.utils.is_complex(factors[0]) and factors[0].dtype == torch.complex32:
                factors = [f.to(torch.complex64) for f in factors]
            
            self.tensor_decomposer.init = tl.tenalg.multi_mode_dot(matrix, factors, transpose=True), factors
            
        try:
            _, factors = self.tensor_decomposer.decompose(matrix)
            torch.cuda.empty_cache() #keras.backend.clear_session()
        except Exception as e:
            if self.verbose:
                print(f"Tucker decomposition failed with warm start, trying again with SVD init: {str(e)}")
            # lets try again 
            try:
                matrix = matrix + 1e-8 * torch.randn_like(matrix, dtype=matrix.dtype)  # Add noise for stability
                _, factors = self.tensor_decomposer.decompose(matrix, init='svd', n_iter_max=self.n_iter_max * 2, svd_type='randomized_svd') # try again
            except Exception as e:
                raise e
        torch.cuda.empty_cache() #keras.backend.clear_session()
        
        # Convert factors to half precision if mixed precision is enabled
        factors = [f.to(original_dtype) for f in factors]
        
        return factors
    
    @tdecomp.utils.no_grad
    def transform(self, proj_tensor: list[TensorLike], full_rank_grad: TensorLike) -> TensorLike:
        with record_function("### TENSOR_GRAD_TRANSFORM"):
            return tl.tenalg.multi_mode_dot(full_rank_grad, proj_tensor, transpose=True)

    @tdecomp.utils.no_grad
    def inverse_transform(self, proj_tensor: TensorLike, x: TensorLike, output_buffer: Optional[torch.Tensor] = None, alpha=1.0) -> TensorLike:
        with record_function("### TENSOR_GRAD_INV_TRANSFORM"):
            if output_buffer is None:
                return tl.tenalg.multi_mode_dot(x, proj_tensor)
            else:
                result = multi_mode_dot(x, proj_tensor)
                output_buffer.add_(result, alpha=alpha)
                return output_buffer