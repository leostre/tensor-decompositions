import pytest 

import torch

from tdecomp.tensor.tucker import DECOMPOSERS
from tdecomp._base import TensorDecomposer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

X = torch.randn(100, 100, 100, device=DEVICE)
RTOL = 1e-3

@pytest.mark.parametrize('name', DECOMPOSERS.keys())
def test_tensor_decomposer(name):
    decomposer: TensorDecomposer = DECOMPOSERS[name]()
    approximation = decomposer.decompose(X)
    error = decomposer.get_approximation_error(X, *approximation, relative=True)
    assert error < RTOL, f'{name} returns approximation violating rtol: error = {error} & relative tolerance = {RTOL}'
