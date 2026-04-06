import math
import torch
from torch.autograd.profiler import record_function
from typing import Optional

import tdecomp
from tdecomp.grad_proj.tensorgrad.config import SparseType
from tdecomp.grad_proj.tensorgrad.projectors.abstract_sparce_projector import AbstractSparceProjector
from tdecomp.grad_proj.tensorgrad.projectors.update_gap_scheduler import UpdateGapScheduler
from tdecomp.types import TensorLike
import tensorly as tl

def mode_unfolding_norms(tensor: TensorLike, mode: int) -> TensorLike:
    """
    Returns L2 norm of each 'row' in the mode-unfolding for dimension=mode.
    If tensor.shape= (D0, D1, ..., Dn), the unfolding w.r.t. mode is shape
    (Dm, product_of_other_dims). We get a vector of length Dm with norms.
    """
    unfolded = tl.unfold(tensor, mode)
    row_norms = tl.norm(unfolded, order=2, axis=1)
    return row_norms


class TensorGradSparseProjector(AbstractSparceProjector):
    """
    N-D projector that uses dimension-wise structured sparsity
    instead of Tucker (like TensorGradLowRankProjector). For each dimension i,
    we sample a subset of indices according to norms (topk, probability, etc.).
    """
    def __init__(
        self,
        sparse_ratio: list[float] | float = 0.25,
        sparse_type: SparseType = "topk",
        verbose: bool = False,
        update_gap_scheduler: UpdateGapScheduler = UpdateGapScheduler(100, 1000),
        scale: float = 1.0,
        warm_restart: bool = False,
        n_iter_max: int = 10,
        scale_by_mask_ratio: bool = False,
    ):
        self.sparse_ratio = sparse_ratio
        self.sparse_type: SparseType = sparse_type
        self.verbose = verbose
        self.update_gap_scheduler = update_gap_scheduler
        self.scale = scale
        self.warm_restart = warm_restart
        self.n_iter_max = n_iter_max
        self.scale_by_mask_ratio = scale_by_mask_ratio
        self.scale_factor = 1.0 * self.scale  # will be updated if scale_by_mask_ratio=True

        # store one mask per dimension => a list of length = tensor.ndim
        self.masks = None
        self._orig_shape = None
        self._last_iter = -1

        print(f"TensorGradSparseProjector initialized with sparse_ratio={self.sparse_ratio}, sparse_type={self.sparse_type}, scale_by_mask_ratio={self.scale_by_mask_ratio}")
        
        
    def should_update_projector(self, iter: int) -> bool:
        return self.update_gap_scheduler.should_update(iter)

    @tdecomp.utils.no_grad
    def project(self, full_rank_grad: TensorLike, iteration: int) -> TensorLike:
        with record_function("### TENSOR_SPARSE_PROJECT_FORWARD"):
            if self._orig_shape is None:
                self._orig_shape = tl.shape(full_rank_grad)

            # Only update masks if necessary
            if (self.masks is None) or self.should_update_projector(iteration):
                self.masks = self._build_masks(full_rank_grad)
    
            # Now transform => produce smaller sub-tensor
            smaller = self._transform(full_rank_grad)
            return smaller

    @tdecomp.utils.no_grad
    def project_back(self, small_grad: TensorLike, output_buffer: Optional[torch.Tensor] =None, alpha=1.0, accumulate=False) -> TensorLike:
        with record_function("### TENSOR_SPARSE_PROJECT_BACK"):
            # Create a temporary buffer if none provided
            if output_buffer is None:
                output_buffer = tl.zeros(self._orig_shape, **tl.context(small_grad))
            full = self._inverse_transform(small_grad, output_buffer, alpha=alpha, accumulate=accumulate)
            return full
    
    def _build_masks(self, tensor: TensorLike) -> list[TensorLike]:
        """
        Build dimension-wise masks for tensor sparsity. Every mask show selected rows.
        """
        # Ensure sparse_ratio is a list of floats, one per dimension.
        if isinstance(self.sparse_ratio, float):
        # make multipliers for each dimension
            non_one_dims = sum(1 for d in tl.shape(tensor) if d != 1)
            even_distributed_ratio = self.sparse_ratio ** (1/non_one_dims)
            # Handle dimensions of size 1 separately
            ratio_list = []
            for d in tl.shape(tensor):
                if d == 1:
                    # Keep ratio 1 for dimensions of size 1
                    ratio_list.append(1)
                else:
                    # Round up for non-1 dimensions
                    ratio_list.append(even_distributed_ratio)
        elif isinstance(self.sparse_ratio, list):
            # if list of floats, convert to multipliers
            if all(isinstance(r, float) and 0 < r < 1 for r in self.sparse_ratio):
                ratio_list = [r ** (1/tl.ndim(tensor))
                                    for r in self.sparse_ratio]
            else:
                ratio_list = self.sparse_ratio
        else:
                raise ValueError(f"Invalid sparse_ratio: {self.sparse_ratio}")

        masks = []
        for mode in range(tl.ndim(tensor)):
            ratio = ratio_list[mode] if mode < len(ratio_list) else ratio_list[-1]
            dim_size = tl.shape(tensor)[mode]

            if self.sparse_type.lower() not in ['randomk', 'randk']:
                # Compute norms directly using mode_unfolding_norms function
                row_norms = mode_unfolding_norms(tensor, mode)
            else:
                row_norms = None

            k = max(1, int(ratio * dim_size + 1))

            mask, _ = self._create_sparse_mask(row_norms, self.sparse_type.lower(), k, tl.context(tensor)) #type: ignore

            # Keep mask on GPU
            masks.append(mask)
            if row_norms is not None:
                del row_norms

        if getattr(self, "scale_by_mask_ratio", False):
            # Compute overall scaling factor more efficiently
            scale_factor = 1.0
            for m in masks:
                kept = tl.sum(m)
                total = tdecomp.utils.numel(m)
                scale_factor *= total / (kept + 1e-8)
            scale_factor = math.sqrt(scale_factor)
            self.scale_factor = scale_factor * self.scale
            self.scale_by_mask_ratio = False
            print(
                f"[TensorGradSparseProjector] Setting the scale factor: {self.scale_factor}")

        if self.verbose:
            kept = [tl.sum(m) for m in masks]
            print(
                f"[TensorGradSparseProjector] Recomputed masks => dims kept: {kept}")

        return masks
    
    def _transform(self, x: TensorLike) -> TensorLike:
        """
        Use masks to select elements from each dimension.
        The result is smaller: shape= (K0, K1, ..., Kn).
        """
        res = x
        for mode, mask in enumerate(self.masks):
            # Create a slice object for this dimension
            slice_obj = [tl.index[None]] * tl.ndim(x)
            slice_obj = tuple(tl.index_update(slice_obj, tl.index[mode], mask))
            # Apply the mask to this dimension
            res = res[slice_obj]

        return res


    def _inverse_transform(self, small_x: TensorLike, output_buffer: Optional[torch.Tensor] = None, alpha=1.0, accumulate=False):
        """
        Start from zero[full_shape], fill in the sub-tensor 'small_x'
        dimension by dimension.
        """
        res = small_x * self.scale_factor
        nd = len(self._orig_shape)
            
        # Create or use the provided buffer
        if output_buffer is None:
            output_buffer = torch.zeros(self._orig_shape, **tl.context(res))

                
        # Process each mode in reverse order
        for mode in reversed(range(nd)):
            # Mask is already on GPU
            mask = self.masks[mode]
                
            # Create slice object for the current mode
            slice_obj = [tl.index[None]] * nd
            slice_obj = tuple(tl.index_update(slice_obj, tl.index[mode], mask))
                
            # Create a temporary tensor of the correct shape
            shape_expanded = list(tl.shape(res))
            shape_expanded = tl.index_update(shape_expanded, tl.index[mode], self._orig_shape[mode])
            bigger = tl.zeros(shape_expanded, **tl.context(res))
                
            # Fill in the values
            bigger = tl.index_update(bigger, slice_obj, res)
            res = bigger
                
        # Final update to the buffer
        if accumulate:
            output_buffer.add_(res, alpha=alpha)
        else:
            output_buffer.copy_(res * alpha)
                
        return output_buffer