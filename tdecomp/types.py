from typing import Any, Callable, Literal, TypeAlias, Union
import tensorly as tl

tl.set_backend('pytorch') #TODO think about place of it
# import os
#-----------
# os.environ["KERAS_BACKEND"] = "torch" #TODO it should be together
# import keras
#-------

TensorLike: TypeAlias = Any
'''Tensorly supports work with different tensor backends (numpy, torch.tensor and so on), 
but it doesnt describe abstract class for it. 
So the tensor can be of `Any` type depending on backend setted in `tl.set_backend` .'''

Number = Union[int, float]
'''Type, widely used for 'rank' typing'''

TensorDecompositionInit: TypeAlias = tuple[TensorLike, list[TensorLike]] | Literal['svd', 'random']
'''Used in iterative tensor decomposition algorithms to determine (start factorization)/(algorithm for start factorization) that will be optimized'''

SVDCallable: TypeAlias = Callable[[TensorLike], tuple[TensorLike, TensorLike, TensorLike]]

BOOL_TYPE = tl.tensor([True]).dtype
COMPLEX64_TYPE = tl.backend.complex64