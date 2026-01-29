import pytest 

import torch

from tdecomp.matrix.decomposer import DECOMPOSERS
from tdecomp._base import DIM_LIM, Decomposer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SQUARE_MATRIX = torch.randn(100, 100, device=DEVICE)
FAT_MATRIX = torch.randn(10, 100, device=DEVICE)
SKINNY_MATRIX = torch.randn(100, 10,device=DEVICE)
BIG_SKINNY_MATRIX = torch.randn(DIM_LIM + 1, 10,device=DEVICE)
RTOL = 1e-5

@pytest.mark.parametrize('X', [SQUARE_MATRIX, FAT_MATRIX, SKINNY_MATRIX, BIG_SKINNY_MATRIX], ids=["square", "fat", "skinny", "big_skinny"])
@pytest.mark.parametrize('name', DECOMPOSERS.keys())
def test_decomposer_decompose_square_matrix(name, X):
    decomposer: Decomposer = DECOMPOSERS[name](rank=min(X.size()))
    dec_result = decomposer.decompose(X, 100)
    error = decomposer.get_approximation_error(X, *dec_result, relative=True)
    assert error < RTOL, f'{name} returns approximation violating rtol: error = {error} & relative tolerance = {RTOL}'