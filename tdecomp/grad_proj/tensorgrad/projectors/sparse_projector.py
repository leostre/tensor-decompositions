from typing import Literal

import tdecomp
from tdecomp.grad_proj.tensorgrad.projectors.update_gap_scheduler import UpdateGapScheduler
from tdecomp.types import TensorLike
import tensorly as tl

import tdecomp.types

class GaLoreSparseProjector:
    """
    A sparse version of GaLore that uses row/column sampling
    instead of low-rank SVD. It follows the same signature
    and 'proj_type' logic as GaLoreProjector.
    """
    def __init__(
        self,
        sparse_ratio: float = 0.25,
        sparse_type: Literal['topk', 'randk', 'randomk', 'probablility'] = "topk",
        verbose: bool = False,
        update_gap_scheduler: UpdateGapScheduler = UpdateGapScheduler(100, 1000),
        scale: float = 1.0,
        proj_type: Literal['std', 'right', 'left', 'reverse_std'] = 'std',
        activation_checkpoint: bool = False,
    ):
        self.sparse_ratio = sparse_ratio
        '''fraction of rows (or columns) to keep'''
        self.sparse_type = sparse_type
        '''e.g. 'topk', 'randK', 'probability' - how to finnaly chose random columns or rows by their importances'''
        self.verbose = verbose
        self.update_gap_scheduler = update_gap_scheduler
        self.scale = scale
        '''Used in project back as scale factor'''
        self.proj_type = proj_type
        self.activation_checkpoint = activation_checkpoint
        self._mask = None

        # Remember shape for project_back
        self._orig_shape = None
        # Keep track of iteration for update
        self._last_iter = -1

    def project(self, full_rank_grad: TensorLike, iteration: int) -> TensorLike:
        """
        Equivalent to GaLoreProjector.project, but uses row/column sampling
        to produce a smaller tensor from 'full_rank_grad'.
        """
        # store shape for reconstruction
        if self._orig_shape is None:
            self._orig_shape = tl.shape(full_rank_grad)

        grad_2d = full_rank_grad  # we assume 2D here

        # Check if we need to update the mask(s)
        # (only do so every update_proj_gap steps)
        if self.update_gap_scheduler.should_update(iteration) or self._mask is None:
            self._update_masks(grad_2d)

        # Now apply the correct sub-selection
        if self.proj_type == 'std':
            if tl.shape(grad_2d)[0] >= tl.shape(grad_2d)[1]:
                # right: sample columns
                mask = self._mask
                return grad_2d[:, mask]
            else:
                # left: sample rows
                mask = self._mask
                return grad_2d[mask, :]

        elif self.proj_type == 'reverse_std':
            if tl.shape(grad_2d)[0] >= tl.shape(grad_2d)[1]:
                # left: sample rows
                mask = self._mask
                return grad_2d[mask, :]
            else:
                # right: sample columns
                mask = self._mask
                return grad_2d[:, mask]

        elif self.proj_type == 'right':
            # always columns
            mask = self._mask
            return grad_2d[:, mask]

        elif self.proj_type == 'left':
            # always rows
            mask = self._mask
            return grad_2d[mask, :]

        else:
            raise ValueError(f"Unknown proj_type={self.proj_type}")

    def project_back(self, low_rank_grad: TensorLike) -> TensorLike:
        """
        Re-inject the smaller tensor into the original shape, placing zeros
        in the unselected positions. Then multiply by scale.
        """
        grad_2d = tl.zeros(self._orig_shape, **tl.context(low_rank_grad))

        if self.proj_type == 'std':
            if tl.shape(grad_2d)[0] >= tl.shape(grad_2d)[1]:
                # we used columns
                mask = self._mask
                tl.index_update(grad_2d, tl.index[:, mask], low_rank_grad)
            else:
                # we used rows
                mask = self._mask
                tl.index_update(grad_2d, tl.index[mask, :], low_rank_grad)

        elif self.proj_type == 'reverse_std':
            if tl.shape(grad_2d)[0] >= tl.shape(grad_2d)[1]:
                # we used rows
                mask = self._mask
                tl.index_update(grad_2d, tl.index[mask, :], low_rank_grad)
            else:
                # columns
                mask = self._mask
                tl.index_update(grad_2d, tl.index[:, mask], low_rank_grad)

        elif self.proj_type == 'right':
            # columns
            mask = self._mask
            tl.index_update(grad_2d, tl.index[:, mask], low_rank_grad)

        elif self.proj_type == 'left':
            # rows
            mask = self._mask
            tl.index_update(grad_2d, tl.index[mask, :], low_rank_grad)

        else:
            raise ValueError(f"Unknown proj_type={self.proj_type}")

        return grad_2d * self.scale

    def _update_masks(self, grad_2d: TensorLike) -> None:
        '''Update 1D mask that represent randomly chosen rows or columns (projection)'''
        self._last_iter += 1
        if self.proj_type in ['std', 'reverse_std', 'left', 'right']:
            # Decide whether to do row or column sampling based on the shape + proj_type
            # We'll just store in self._mask
            do_rows = False
            if self.proj_type == 'left':
                do_rows = True
            elif self.proj_type == 'right':
                do_rows = False
            elif self.proj_type == 'std':
                do_rows = (tl.shape(grad_2d)[0] < tl.shape(grad_2d)[1])
            elif self.proj_type == 'reverse_std':
                do_rows = (tl.shape(grad_2d)[0] >= tl.shape(grad_2d)[1])

            self._mask = self._select_mask_1d(grad_2d, do_rows)

        else:
            raise ValueError(f"Unknown proj_type={self.proj_type}")

    def _select_mask_1d(self, grad_2d: TensorLike, do_rows=True) -> TensorLike:
        """
        Produce a 1D boolean mask along dimension 0 (rows) if do_rows=True,
        or dimension 1 (columns) if do_rows=False.
        """
        if do_rows:
            norms = tl.norm(grad_2d, order=2, axis=1)  # shape=(nrows,)
            dim_size = tl.shape(grad_2d)[0]
        else:
            norms = tl.norm(grad_2d, order=2, axis=0)  # shape=(ncols,)
            dim_size = tl.shape(grad_2d)[1]

        k = max(1, int(self.sparse_ratio * dim_size))
        
        # pick indices
        if self.sparse_type.lower() == 'topk':
            idxs = tl.argsort(norms, 0)[-k:]

        elif self.sparse_type.lower() in ('randk', 'randomk'):
            idxs = tdecomp.utils.randperm(dim_size, tl.context(grad_2d))[:k]

        elif self.sparse_type.lower() == 'probability':
            idxs = tdecomp.utils.multinomial(norms, k, tl.context(grad_2d))

        else:
            raise ValueError(f"Unknown sparse_type={self.sparse_type}")
        
        mask = tl.zeros(dim_size, **tl.context(grad_2d))
        mask = tl.tensor(mask, dtype=tdecomp.types.BOOL_TYPE)
        tl.index_update(mask, tl.index[idxs], True)
        return mask