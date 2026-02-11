from typing import Any, Callable, Literal, Union
import tensorly as tl

type TensorLike = Any
'''Tensorly supports work with different tensor backends (numpy, torch.tensor and so on), 
but it doesnt describe abstract class for it. 
So the tensor can be of `Any` type depending on backend setted in `tl.set_backend` .'''

Number = Union[int, float]
'''Type, widely used for 'rank' typing'''

type TensorDecompositionInit = tuple[TensorLike, list[TensorLike]] | Literal['svd', 'random']
'''Used in iterative tensor decomposition algorithms to determine (start factorization)/(algorithm for start factorization) that will be optimized'''

type SVDCallable = Callable[[TensorLike], tuple[TensorLike, TensorLike, TensorLike]]

BOOL_TYPE = tl.tensor([True]).dtype
COMPLEX64_TYPE = tl.backend.complex64