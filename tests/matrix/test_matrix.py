import pytest 

import torch

from tdecomp.matrix.decomposer import DECOMPOSERS
from tdecomp._base import Decomposer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RTOL = 1e-4

Xs = [
    torch.randn(100, 100, device=DEVICE),
    torch.randn(200, 100, device=DEVICE),
    torch.randn(100, 200, device=DEVICE),
]

@pytest.mark.parametrize('name', DECOMPOSERS.keys())
def test_decomposer_relative_error(name):
    for X in Xs:
        decomposer: Decomposer = DECOMPOSERS[name](rank=min(X.size()))
        dec_result = decomposer.decompose(X, 100)
        error = decomposer.get_approximation_error(X, *dec_result, relative=True)
        assert error < RTOL, f'{name} returns approximation violating rtol: error = {error} & relative tolerance = {RTOL}. Size: {X.size()}'
