import torch
from tdecomp.matrix.random_projections import ortho
import tensorly as tl


def test_ortho_generator():
    tl.set_backend("pytorch")
    orto_tl = ortho(5, 3)
    assert torch.allclose(tl.matmul(tl.transpose(orto_tl), orto_tl), torch.eye(3), atol=1e-5)
