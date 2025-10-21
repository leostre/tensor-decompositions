import pytest 

import torch

from tdecomp.matrix.decomposer import DECOMPOSERS
from tdecomp._base import Decomposer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

X = torch.randn(100, 100, device=DEVICE)
RTOL = 1e-4

@pytest.mark.parametrize('name', DECOMPOSERS.keys())
def test_decomposer_relative_error(name):
    decomposer: Decomposer = DECOMPOSERS[name](rank=min(X.size()))
    dec_result = decomposer.decompose(X, 100)
    error = decomposer.get_approximation_error(X, *dec_result, relative=True)
    assert error < RTOL, f'{name} returns approximation violating rtol: error = {error} & relative tolerance = {RTOL}'