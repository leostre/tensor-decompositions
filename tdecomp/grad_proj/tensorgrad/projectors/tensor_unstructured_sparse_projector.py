import math
from typing import Optional
import torch
from torch.autograd.profiler import record_function

import tdecomp
from tdecomp.grad_proj.tensorgrad.config import SparseType
from tdecomp.grad_proj.tensorgrad.projectors.abstract_sparce_projector import AbstractSparceProjector
from tdecomp.grad_proj.tensorgrad.projectors.update_gap_scheduler import UpdateGapScheduler
from tdecomp.types import TensorLike
import tensorly as tl


class TensorGradUnstructuredProjector(AbstractSparceProjector):
    """
    N-D projector using unstructured (element-wise) sparsity,
    storing only the values at specified indices.
    """
    def __init__(
        self,
        sparse_ratio: float = 0.25,
        sparse_type: SparseType = "randk",
        verbose: bool = False,
        update_gap_scheduler = UpdateGapScheduler(100, 1000),
        scale: float = 1.0,
        proj_type: str = "std",  # for naming consistency
        warm_restart: bool = False,
        n_iter_max: int = 10,
        scale_by_mask_ratio: bool = False,
    ):
        """
        Parameters:
        -----------
        sparse_ratio: float
            Fraction of elements to keep (if in [0,1]). For topk, that means
            the top k = sparse_ratio * (tensor.numel()). 
        sparse_type: str
            'topk', 'probability', 'randk' (randomk). 
        update_gap_scheduler: Instance of UpdateGapScheduler
            Controls when to update the projection indices.
        scale_by_mask_ratio: bool
            If True, will scale up the recovered tensor by the ratio 
            (total_elem / kept_elem) to preserve overall magnitude.
        """
        self.sparse_ratio = sparse_ratio
        self.sparse_type = sparse_type
        self.verbose = verbose
        self.update_gap_scheduler = update_gap_scheduler
        self.base_scale = scale
        self.proj_type = proj_type
        self.warm_restart = warm_restart
        self.n_iter_max = n_iter_max
        self.scale_by_mask_ratio = scale_by_mask_ratio
        self.scale_factor = scale  # Will be updated if scale_by_mask_ratio=True
        self.device = None
        self.should_update = False
        self._orig_shape = None
        self._indices = None
        self._last_iter = -1
        
        print(f"Update gap scheduler: {self.update_gap_scheduler}")
        print(f"UnstructuredSparseProjector initialized with sparse_ratio={self.sparse_ratio}, sparse_type={self.sparse_type}, scale_by_mask_ratio={self.scale_by_mask_ratio}")
        
    def should_update_projector(self, iteration: int):
        """Check if the projector indices should be updated in this iteration"""
        if self._indices is None:
            self.should_update = True
        if self.update_gap_scheduler is not None:
            self.should_update = self.update_gap_scheduler.should_update(iteration)
        return self.should_update

    @tdecomp.utils.no_grad
    def project(self, full_grad: TensorLike, iteration: int) -> TensorLike:
        with record_function("### UNSTRUCTURED_SPARSE_PROJECT_FORWARD"):
            if self._orig_shape is None:
                self._orig_shape = tl.shape(full_grad)

            # Only update indices if necessary
            if (self._indices is None) or self.should_update_projector(iteration):
                self._build_indices(full_grad)

            # Just return the values at the selected indices
            flat = full_grad.view(-1)
            result = flat[self._indices]
                
            return result
    
    @tdecomp.utils.no_grad
    def project_back(
        self,
        small_grad: TensorLike,
        output_buffer: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        accumulate: bool = False
    ) -> TensorLike:
        """
        Back-project a sparse vector into `output_buffer`, either overwriting
        (accumulate=False) or adding to existing contents (accumulate=True).
        
        Args:
            small_grad: Tensor of values for the selected indices
            output_buffer: Pre-allocated buffer to write into or None to create a new tensor
            alpha: Scaling factor to apply to the values
            accumulate: If True, add to buffer values; if False, overwrite
            
        Returns:
            The output_buffer or a new tensor if output_buffer is None
        """
        with record_function("### UNSTRUCTURED_SPARSE_PROJECT_BACK"):
            # If no buffer provided, create a new one
            if output_buffer is None:
                # Create a zero tensor with the original shape
                output_buffer = tl.zeros(self._orig_shape, **tl.context(small_grad))
                # For new buffers, we always overwrite (accumulate flag is ignored)
                accumulate = False
            else:
                # Ensure buffer shape matches original shape
                assert tl.shape(output_buffer) == self._orig_shape, f"Buffer shape {output_buffer.shape} doesn't match original shape {self._orig_shape}"                
                # Ensure consistent dtype between small_grad and output_buffer
                if tl.context(small_grad)["dtype"] != tl.context(output_buffer)["dtype"]:
                    if self.verbose:
                        print(f"Converting small_grad from {tl.context(small_grad)["dtype"]} to {tl.context(output_buffer)["dtype"]} for consistency")
                    small_grad = tl.tensor(small_grad, **tl.context(output_buffer))
            
            # Scale values once (combining alpha and scale_factor)
            # Convert to the same dtype as the output buffer to avoid dtype mismatch in scatter_
            vals = (small_grad * (self.scale_factor * alpha))
            
            # Store original shape to reshape back at the end
            original_shape = tl.shape(output_buffer)
            
            # Get flattened version of buffer - try to use view first
            if output_buffer.is_contiguous():
                flat = output_buffer.view(-1)
            else:
                # Need to reshape if not contiguous, but this creates a new tensor
                # We'll need to copy back to the original buffer at the end
                flat = output_buffer.reshape(-1)
            
            # Check if we're using ComplexHalf dtype which doesn't support scatter operations
            is_complex_half = tl.context(flat)["dtype"] == torch.complex32
            
            # If using ComplexHalf, temporarily convert to a supported type
            if is_complex_half:
                temp_buffer = flat.to(torch.complex64)
                temp_vals = vals.to(torch.complex64)
                temp_indices = self._indices
                
                if not accumulate:
                    # Overwrite mode: clear only the positions we touch
                    temp_buffer.scatter_(0, temp_indices, temp_vals)
                else:
                    # Accumulate mode: add to whatever is already in the buffer
                    temp_buffer.scatter_add_(0, temp_indices, temp_vals)
                
                # Convert back to ComplexHalf
                flat.copy_(temp_buffer.to(torch.complex32))
            else:
                if not accumulate:
                    # Overwrite mode: clear only the positions we touch
                    flat.scatter_(0, self._indices, vals)
                else:
                    # Accumulate mode: add to whatever is already in the buffer
                    flat.scatter_add_(0, self._indices, vals)
            
            # If we had to reshape, copy the modified flat tensor back to output_buffer
            if not output_buffer.is_contiguous():
                # Reshape flat back to original shape
                modified = flat.reshape(original_shape)
                # Copy back to the original buffer
                output_buffer.copy_(modified)
                
            return output_buffer

    def _build_indices(self, x: TensorLike):
        """
        Build a 1D LongTensor of indices to keep in the flattened tensor.
        """
        flat = x.view(-1)
        numel = tl.shape(flat)[0]
        k = max(1, int(self.sparse_ratio * numel))

        _, idx = self._create_sparse_mask(tl.abs(flat), self.sparse_type.lower(), k, tl.context(x))

        # Sort for better memory locality
        self._indices = tl.sort(idx, 0)
            
        del idx
        torch.cuda.empty_cache()

        # Possibly compute scaling factor to preserve total norm
        if self.scale_by_mask_ratio:
            # sqrt to preserve L2 norm
            self.scale_factor = self.base_scale * math.sqrt(numel / k)
            # Only do it once
            self.scale_by_mask_ratio = False
            if self.verbose:
                print(f"Set scale factor to {self.scale_factor:.4f}")
