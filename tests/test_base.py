import pytest
import numpy as np
import torch
import tensorly as tl

from tdecomp._base import Decomposer, TensorDecomposer, _conditioning, _need_t
from tdecomp.matrix.decomposer import RandomizedSVD 

SAMPLE_TENSOR = torch.tensor([[1, 0], [0, 1]]) * 1.0

def test_need_t():

    @_need_t
    def decomposition_function_stub(self, t):
        return t, SAMPLE_TENSOR
    
    initial_tensor = torch.arange(24).reshape(2, 12)
    factors = decomposition_function_stub(None, initial_tensor)
    assert (factors[0] == initial_tensor).all(), "Tensor shouldn't be transposed if m < n"

    initial_tall_tensor = torch.arange(24).reshape(12, 2)
    factors2 = decomposition_function_stub(None, initial_tall_tensor)
    assert (factors2[1] == initial_tall_tensor).all(), "Initial matrix should be transposed"
    assert (factors2[0] == SAMPLE_TENSOR.T).all(), "Returned matrix should be transposed"


class TestConditioning:
    U = tl.eye(2)
    S = tl.eye(2)
    Vh = torch.tensor([[1, 0, 0], [0, 3, 0], [0, 0, 5], [8, 2, 9]]) * 0.1 # 4x3
    Vh_with_same_shape = torch.tensor([[1, 0, 0, 0], [0, 3, 0, 0], [0, 0, 5, 0]]) * 0.1 #3x4
    W = torch.tensor([[0, 1, 0, 0], [0, 0, 4, 0], [0, 0, 0, 2]]) * 0.1 # 3x4
    _conditioner = None

    @_conditioning
    def decomposition_function_stub(self, t, *args, **kwargs):
        return self.U, self.S, self.Vh
    
    @_conditioning
    def decomposition_function_stub_same_shape(self, t, *args, **kwargs):
        return self.U, self.S, self.Vh_with_same_shape

    def test_without_passed_conditioner(self):
        factors = self.decomposition_function_stub(self.W)
        assert (factors[0] == self.U).all()
        assert (factors[1] == self.S).all()
        assert (factors[2] == self.Vh).all()

    def test_with_passed_conditioner_2D_non_square(self):
        conditioner = self.W.T #4x3
        inv_conditioner = torch.linalg.pinv(conditioner) #3x4
        factors = self.decomposition_function_stub(self.W, conditioner=conditioner)
        assert (factors[0] == self.U).all()
        assert (factors[1] == self.S).all()
        assert torch.allclose(factors[2], tl.matmul(self.Vh, inv_conditioner))

    def test_with_passed_conditioner_2D_square(self):
        conditioner = torch.diag(torch.tensor([1, 2, 5, 8.0]))
        inv_conditioner = torch.linalg.pinv(conditioner) #4x4
        factors = self.decomposition_function_stub_same_shape(self.W, conditioner=conditioner)
        assert (factors[0] == self.U).all()
        assert (factors[1] == self.S).all()
        assert torch.allclose(factors[2], tl.matmul(self.Vh_with_same_shape, inv_conditioner))

    def test_with_passed_conditioner_1D(self):
        conditioner = torch.tensor([5, 8, 10, 4]) #x4
        inv_conditioner = 1 / (conditioner) #x4
        factors = self.decomposition_function_stub_same_shape(self.W, conditioner=conditioner)
        assert (factors[0] == self.U).all()
        assert (factors[1] == self.S).all()
        assert torch.allclose(factors[2], tl.matmul(self.Vh_with_same_shape, tl.diag(inv_conditioner)))

    def test_singular_conditioner_2D_raise_error(self):
        conditioner = torch.arange(16.0).reshape(4, 4)
        inv_conditioner = torch.linalg.pinv(conditioner) #4x4
        with pytest.raises(RuntimeError) as e:
            factors = self.decomposition_function_stub(self.W, conditioner=conditioner)

    def test_singular_conditioner_1D_returns_nan(self):
        conditioner = torch.tensor([1, 2, 3, 0])
        factors = self.decomposition_function_stub_same_shape(self.W, conditioner=conditioner)
        assert factors[2][0][3].isnan()

def test_compose():
    U, S, Vh = tl.svd_interface(SAMPLE_TENSOR)
    # print(len(tensors))
    assert torch.allclose(RandomizedSVD().compose(U, S, Vh), SAMPLE_TENSOR)
    assert torch.allclose(RandomizedSVD().compose(U * S, Vh), SAMPLE_TENSOR)