from functools import partial

from typing import *

import tdecomp
from tdecomp.types import Number, TensorLike
from tdecomp._base import Decomposer, _need_t
from tdecomp.matrix.random_projections import ProjectorGenerator
from tdecomp.matrix.importance_generators import ColumnRowImportancesGenerator
import tensorly as tl

__all__ = [
    'SVDDecomposition',
    'RandomizedSVD',
    'TwoSidedRandomSVD',
    'CURDecomposition'
]

class SVDDecomposition(Decomposer):
    def _decompose(self, X: TensorLike, rank) -> tuple[TensorLike, TensorLike, TensorLike]:
        """Standart SVD decomposition, realization depends on various backends.  
        Result is non-determenistic, sign of U and V can change in columns together.

        Args:
            W: matrix to decompose
        Returns:
            U, S, Vt: decomposition
        """
        return tl.truncated_svd(X, n_eigenvecs=min(tl.shape(X)))


class RandomizedSVD(Decomposer):
    """
    https://arxiv.org/pdf/2404.09276
    """

    def __init__(self, rank: Optional[Number] = None, power: int = 3,
                 distortion_factor: float = 0.6, 
                 random_init: ProjectorGenerator = ProjectorGenerator.normal):
        super().__init__(rank, distortion_factor, random_init)
        self.power = power

    def estimate_stable_rank(self, W: TensorLike) -> int:
        svals_squared = tl.truncated_svd(W, n_eigenvecs=min(tl.shape(W)))[1] ** 2 #NOTE вычисляется полный SVD - нет смысла в этом, если вдруг не передадим rank в decompose()
        stable_rank = (tl.sum(svals_squared) / tl.max(svals_squared))
        return max(1, min(min(tl.shape(W)), int(stable_rank * (1 / self.distortion_factor))))
    
    @_need_t
    def _decompose_big(self, X: TensorLike, rank: int) -> tuple[TensorLike, TensorLike, TensorLike]:
        P = self.random_init.value(rank, tl.shape(X)[-2], tl.context(X))
        G = tl.matmul(P, tl.matmul(X, tl.matmul(tl.transpose(X), tl.transpose(P))))
        Q, _ = tl.qr(
            tl.transpose(tl.matmul(G ** self.power, tl.matmul(P, X))),
            mode='reduced')
        B = tl.matmul(X, Q)
        U, S, Vh = tl.truncated_svd(B, n_eigenvecs=min(tl.shape(B)))
        return U, S, tl.matmul(Vh, tl.transpose(Q))
        
    @_need_t
    def _decompose(self, X: TensorLike, rank: int) -> tuple[TensorLike, TensorLike, TensorLike]:
        G = tl.matmul(X, tl.transpose(X))
        P = self.random_init.value(tl.shape(X)[-1], rank, tl.context(X))
        Q, _ = tl.qr(tl.matmul(G ** self.power, tl.matmul(X, P)), mode='reduced')
        B = tl.matmul(tl.transpose(Q), X)
        U, S, Vh = tl.truncated_svd(B, n_eigenvecs=min(tl.shape(B)))
        return tl.matmul(Q, U), S, Vh


class TwoSidedRandomSVD(RandomizedSVD):
    """
    Randomized Two-Sided SVD with explicit rank parameter support
    https://scispace.com/pdf/randomized-algorithms-for-computation-of-tucker-1stsnpusvv.pdf
    """
    def __init__(self, rank: Optional[int] = None, distortion_factor: float = 0.6, 
                 random_init: ProjectorGenerator = ProjectorGenerator.normal):
        super().__init__(rank=rank, distortion_factor=distortion_factor, random_init=random_init)
        if random_init == ProjectorGenerator.lean_walsh and rank is not None:
            if not (rank > 0 and (rank & (rank - 1) == 0)):
                raise ValueError(f"For lean_walsh, rank must be power of 2, got {rank}")
    
    def _decompose(self, X: TensorLike, rank: int) -> Tuple[TensorLike, TensorLike, TensorLike]:
        I, J = tl.shape(X)[-2], tl.shape(X)[-1]
        random_gen: partial[TensorLike] = self.random_init.value
        Omega1 = random_gen(J, rank, tl.context(X))
        Omega2 = random_gen(I, rank, tl.context(X))
            
        Y1 = tl.matmul(X, Omega1)
        Y2 = tl.matmul(tl.transpose(X), Omega2)
            
        Q1, _ = tl.qr(Y1, mode='reduced')
        Q2, _ = tl.qr(Y2, mode='reduced')
            
        B = tl.matmul(tl.transpose(Q1), tl.matmul(X, Q2))
            
        U_bar, S, Vh_bar = tl.truncated_svd(B, n_eigenvecs=min(tl.shape(B)))
        U = tl.matmul(Q1, U_bar)
        Vh = tl.transpose(tl.matmul(Q2, tl.transpose(Vh_bar)))  
        return U, S, Vh


class CURDecomposition(Decomposer):
    """
    CUR decomposition is a low-rank matrix decomposition method that is based on selecting
    a subset of columns and rows of the original matrix. The method is based on the
    Johnson-Lindenstrauss lemma and is used to approximate the original matrix with a
    low-rank matrix. The CUR decomposition is defined as follows:
    A = C @ U @ R
    where A is the original matrix, C is a subset of columns of A, U is a subset of rows of A,
    and R is a subset of rows of A. The selection of columns and rows is based on the
    probabilities p and q, which are computed based on the norms of the columns and rows of A.
    The selection of columns and rows is done in such a way that the approximation error is minimized.

    Args:
        params: the parameters of the operation
            rank: the rank of the decomposition
            tolerance: the tolerance of the decomposition
            return_samples: whether to return the samples or the decomposition matrices

    """

    def __init__(self, rank: Optional[Number] = None, distortion_factor: float = 0.6, 
                 random_init = ColumnRowImportancesGenerator.l2_norm):
        super().__init__(random_init=random_init, rank=rank, distortion_factor=distortion_factor)
        
    def _decompose(self, X: TensorLike, rank: int) -> Tuple[TensorLike, TensorLike, TensorLike]:
        # create sub matrices for CUR-decompostion
        c, w, r = self.select_rows_cols(X, rank)
        # evaluate pseudoinverse for W - U^-1
        u = tdecomp.utils.pseudo_inverse(w)
        # aprox U using pseudoinverse
        return c, u, r

    def _importance(self, X) -> tuple[TensorLike, TensorLike]:
        col_probs, row_probs = cast(ColumnRowImportancesGenerator, self.random_init).value(X)
        return col_probs, row_probs
    

    def select_rows_cols(self, X: TensorLike, rank: int) -> tuple[TensorLike, TensorLike, TensorLike]:
        # Evaluate norms for columns and rows
        col_probs, row_probs = self._importance(X)

        
        column_indices = tl.sort(tl.argsort(col_probs, 0)[-rank:], 0)
        row_indices = tl.sort(tl.argsort(row_probs, 0)[-rank:], 0)

        C_matrix = X[:, column_indices] 
        R_matrix = X[row_indices, :]
        W_matrix = X[row_indices, :][:, column_indices]

        return C_matrix, W_matrix, R_matrix

    def compose(self, *factors: TensorLike, **kwargs) -> TensorLike:
        C, U, R = factors
        return tl.matmul(C, tl.matmul(U, R))

__local_names = locals() 

DECOMPOSERS: Dict[str, Decomposer]= {
    name: __local_names[name] for name in __all__
}
