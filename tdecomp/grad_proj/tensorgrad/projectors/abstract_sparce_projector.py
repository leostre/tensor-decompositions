import tdecomp
from tdecomp.grad_proj.tensorgrad.config import SparseType
from tdecomp.types import TensorLike
import tensorly as tl


class AbstractSparceProjector:
    def _create_sparse_mask(self, scores: TensorLike, sparse_type: SparseType, k: int, context: dict = {}) -> TensorLike:
        '''Creates mask from given row/column scores/weights/norms with k True values
        
        Returns:
            mask: mask of len(scores)

        '''
        dim_size = tl.shape(scores)
        if sparse_type == 'topk':
            idxs = tdecomp.utils.topk_ids(scores, k)

        elif sparse_type in ('randk', 'randomk'):
            idxs = tdecomp.utils.randperm(dim_size, context)[:k]

        elif sparse_type == 'probability':
            idxs = tdecomp.utils.multinomial(scores, k, context)

        else:
            raise ValueError(f"Unknown sparse_type={sparse_type}")
        
        mask = tdecomp.utils.bool_mask(dim_size, context)
        mask = tl.index_update(mask, tl.index[idxs], True)
        return mask