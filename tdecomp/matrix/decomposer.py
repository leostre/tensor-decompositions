import abc
import torch

class Decomposer(abc.ABC):
    @abc.abstractmethod
    def decompose(self, tensor: torch.Tensor):
        pass
    