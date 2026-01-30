import pytest
import torch
import tdecomp
from tdecomp.matrix.random_projections import Projector, ProjectorGenerator, four_wise_independent_matrix, identity_copies, lean_walsh, ortho, sparse_iid_entries
import tensorly as tl

@pytest.fixture(autouse=True)
def init_tensorly_backend():
    tl.set_backend("pytorch")

def test_ortho_generator():
    orto_tl = ortho(5, 3)
    assert torch.allclose(tl.matmul(tl.transpose(orto_tl), orto_tl), torch.eye(3), atol=1e-5)

def test_norm_generator():
    a = tdecomp.matrix.random_projections.normal(10, 10)
    assert 0 <= abs(tl.mean(a)) < 3
    assert 0 <= abs(tl.mean(a * a)) < 2
    # a = 0
    # for i in range(1000):
    #     a = a + tdecomp.matrix.random_projections.normal(3, 3)
    # assert 0 <= abs(tl.mean(a)) < 3 #WARNING! Error can sum and mean not always near zero, it goes up to positive values
    #assert 500 <= abs(tl.mean(a * a)) <= 1500


def test_sparse_iid_entries_dont_crush():
    assert tl.shape(sparse_iid_entries(3, 4, 3)) == (3, 4)

def test_ssparse_jl_matrix_dont_crush():
    assert tl.shape(sparse_iid_entries(3, 4, 3)) == (3, 4)

def test_four_wise_independent_matrix_dont_crush():
    assert tl.shape(four_wise_independent_matrix(3, 8)) == (3, 8) 

def test_lean_walsh_dont_crush():
    assert tl.shape(lean_walsh(3, 4)) == (3, 4)

def test_identity_copies_dont_crush():
    assert tl.shape(identity_copies(3, 4)) == (3, 4)

def test_projector():
    projector = Projector(ProjectorGenerator.normal)
    H = tl.ones((3, 2))
    assert tl.shape(projector.project(H, 3, side='left')) == (3, 2)
    assert tl.shape(projector.project(H, 3, side='right')) == (3, 3)
    assert tl.shape(projector.project(H, 3, side='right', renew=True)) == (3, 3)
    assert tl.shape(projector.project(H, 4, side='right')) == (3, 4)
