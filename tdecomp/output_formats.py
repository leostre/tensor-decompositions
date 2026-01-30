import abc
import tensorly as tl
from functools import reduce
from tdecomp._base import TensorLike

class _IDecompositionResult(abc.ABC):
    @abc.abstractmethod
    def compose(cls, *tensors) -> TensorLike:
        pass 

    def __init__(self, tensors):
        if isinstance(tensors, dict):
            self.tensors = tensors
        elif isinstance(tensors, (tuple, list)):
            self.tensors = dict(enumerate(tensors))


class LinearDecomposition(_IDecompositionResult):
    @classmethod
    def compose(cls, *tensors) -> TensorLike:
        return reduce(tl.matmul, tensors)


class ModalDecomposition(_IDecompositionResult):
    @classmethod
    def compose(cls, *tensors):
        pass
