from sklearn.random_projection import johnson_lindenstrauss_min_dim
import torch
from functools import wraps, reduce
from typing import *
from abc import ABC, abstractmethod

from fedcore.architecture.utils.misc import filter_kw_universal
# from fedcore.repository.constanst_repository import DIM_SUM_LIM, DIM_LIM

__all__ = [
    'SVDDecomposition',
    'RandomizedSVD',
    'CURDecomposition',
    'RPHOSVDDecomposition',
    'BasicRandomizedSVD',
    'RSTHOSVDDecomposition',
    'RSTDecomposition',
    'DECOMPOSERS',
]

DIM_SUM_LIM = 1024
DIM_LIM = 2048


def _need_t(f):
    """Performs matrix transposition for maximal projection effect"""
    @wraps(f)
    def _wrapper(self, W: torch.Tensor, *args, **kwargs):
            m, n = W.size(-2), W.size(-1)
            _is_transposed = m >= n
            weight = W.t() if _is_transposed else W
            tns = f(self, weight, *args, **kwargs)
            return (
                tns if not _is_transposed
                else tuple(t.t() for t in reversed(tns))
            )
    return _wrapper

def _ortho_gen(x: int, y: int):
    P = torch.empty(x, y)
    torch.nn.init.orthogonal_(P)
    return P

def n_mode_unfolding(X, n):
    """
    Makes n-unfolding of tensor X by given mode n (1-indexed)
    Optimized vectorized implementation

    Args:
        X: torch.Tensor, N-sized tensor with dimension (I_1, I_2, ..., I_N)
        n: int, unfolding mode (1 to N)

    Returns:
        torch.Tensor: Unfolding matrix with dimension (I_n, I_1*...*I_{n-1}*I_{n+1}*...*I_N)
    """
    # Convert to 0-indexed
    n = n - 1
    
    # Get tensor dimensions
    dims = list(X.size())
    N = len(dims)
    
    # Permute tensor to put mode n first
    perm = list(range(N))
    perm.remove(n)
    perm = [n] + perm
    
    # Permute and reshape
    X_permuted = X.permute(perm)
    new_shape = [dims[n]] + [-1]
    X_unfolded = X_permuted.reshape(new_shape)
    
    return X_unfolded

class Decomposer(ABC):
    def decompose(self, W: torch.Tensor, *args, **kwargs):
        if not self._is_big(W):
            return self._decompose(W, *args, **kwargs)
        else:
            return self._decompose_big(W, *args, **kwargs)
        
    def _is_big(self, W: torch.Tensor):
        return sum(W.size()) > DIM_SUM_LIM or any(d > DIM_LIM for d in W.size())

    @abstractmethod
    def _decompose(self, W, *args, **kwargs):
        pass

    def _decompose_big(self, W, *args, **kwargs):
        return self._decompose(W, *args, **kwargs)
    
    def _get_stable_rank(self, W):
        n_samples = max(W.shape)
        min_num_samples = johnson_lindenstrauss_min_dim(n_samples, eps=self.distortion_factors)
        return min(round(min_num_samples), max(W.size()), 1)
    
    def get_approximation_error(self, W, *result_matrices):
        approx = reduce(torch.matmul, result_matrices)
        return torch.linalg.norm(W - approx)


class SVDDecomposition(Decomposer):
    def _decompose(self, W: torch.Tensor) -> tuple:
        """Block Krylov subspace method for computing the SVD of a matrix with a low computational cost.

        Args:
            W: matrix to decompose
        Returns:
            u, s, vt: decomposition

        """
        # Return classic svd decomposition
        return torch.linalg.svd(W, full_matrices=False)
    
    def get_approximation_error(self, W, *result_matrices):
        U, S, Vh = result_matrices
        approx = (U * S) @ Vh
        return torch.linalg.norm(W - approx)


class RandomizedSVD(Decomposer):
    """
    https://arxiv.org/pdf/2404.09276
    """
    _random_gens = {
        'normal': lambda x, y : torch.randn(x, y),
        'ortho' : _ortho_gen
    }

    @filter_kw_universal
    def __init__(self, *, power: int = 3,
                 distortion_factor: float = 0.6, 
                 random_init: str = 'normal'):
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.power = power
        self.distortion_factors = distortion_factor
        self.random_init = random_init
    
    @_need_t
    def _decompose_big(self, X):
        P = torch.randn(self._get_stable_rank(X), X.size(-2), device=X.device, dtype=X.dtype)
        G = P @ X @ (X.T @ P.T)
        Q, _ = torch.linalg.qr(
            (torch.pow(G, self.power) @ (P @ X)).T,
            mode='reduced')
        B = X @ Q
        U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        return U, S, Vh @ Q.T
        
    @_need_t
    def _decompose(self, X):
        G = X @ X.T
        P = torch.randn(X.size(1), self._get_stable_rank(X), device=X.device, dtype=X.dtype)
        Q, _ = torch.linalg.qr(torch.pow(G, self.power) @ X @ P, mode='reduced')
        B = Q.T @ X
        U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        return Q @ U, S, Vh
    
    def get_approximation_error(self, W, *result_matrices):
        U, S, Vh = result_matrices
        approx = (U * S) @ Vh
        return torch.linalg.norm(W - approx)
    

class TensorDecomposer(Decomposer):
    """
    Base class for tensor decomposition algorithms.
    
    This class provides common functionality for tensor decomposition methods such as
    Tucker decomposition variants. It includes shared methods for tensor operations
    like mode-product multiplication, tensor reconstruction from factors, and
    approximation error computation.
    
    Methods:
        _mode_dot: Tensor product along a specified mode
        _reconstruct_tensor: Reconstruct tensor from Tucker decomposition factors
        get_approximation_error: Compute relative approximation error
    """
    
    def _mode_dot(self, tensor, matrix, mode):
        """
        Tensor product along a specified mode.
        
        Args:
            tensor: input tensor
            matrix: matrix to multiply with
            mode: mode number (0-based)
            
        Returns:
            Result of tensor product
        """
        # Move the selected mode to the end
        ndim = tensor.dim()
        perm = list(range(ndim))
        if mode != ndim - 1:
            perm = perm[:mode] + perm[mode + 1:] + [perm[mode]]
        tensor_permuted = tensor.permute(perm)

        # Convert to 2D and multiply
        shape_permuted = tensor_permuted.shape
        tensor_2d = tensor_permuted.reshape(-1, shape_permuted[-1])
        new_tensor_2d = tensor_2d @ matrix.t()

        # Form new shape and restore multidimensionality
        new_shape = shape_permuted[:-1] + (matrix.size(0),)
        new_tensor_permuted = new_tensor_2d.reshape(new_shape)

        # Move the new axis to the original position
        new_perm = list(range(mode)) + [ndim - 1] + list(range(mode, ndim - 1))
        return new_tensor_permuted.permute(new_perm)

    def _reconstruct_tensor(self, core_tensor, factor_matrices):
        """
        Reconstruct tensor from Tucker decomposition: X ≈ [[S; Q^(1), Q^(2), ..., Q^(N)]]
        Optimized implementation
        
        Args:
            core_tensor: core tensor S
            factor_matrices: list of factor matrices [Q^(1), Q^(2), ..., Q^(N)]
            
        Returns:
            reconstructed tensor
        """
        # Tucker decomposition reconstruction: X ≈ [[S; Q^(1), Q^(2), ..., Q^(N)]]
        # This means: X ≈ S ×_1 Q^(1) ×_2 Q^(2) ... ×_N Q^(N)
        
        reconstructed = core_tensor
        
        # Apply factor matrices in sequence using mode_dot
        for i, Q_n in enumerate(factor_matrices):
            # Use mode_dot for proper tensor contraction
            reconstructed = self._mode_dot(reconstructed, Q_n, i)

        return reconstructed

    def get_approximation_error(self, original_tensor: torch.Tensor, *result_matrices) -> torch.Tensor:
        """
        Compute approximation error for tensor decomposition
        
        Args:
            original_tensor: original input tensor
            result_matrices: (core_tensor, factor_matrices) from decomposition
            
        Returns:
            relative approximation error
        """
        if len(result_matrices) != 2:
            raise ValueError("Expected core_tensor and factor_matrices")

        core_tensor, factor_matrices = result_matrices

        # Reconstruct tensor using Tucker decomposition
        reconstructed = self._reconstruct_tensor(core_tensor, factor_matrices)
        
        # Compute relative error
        original_norm = torch.linalg.norm(original_tensor)
        
        return torch.linalg.norm(original_tensor - reconstructed) / original_norm


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

    @filter_kw_universal
    def __init__(self, *, rank: Optional[int] = None, distortion: Union[int, List[int]]):
        self.stable_rank = rank
        self.distortion = distortion

    def get_approximation_error(self, original_tensor, *result_matrices):
        """
        Compute approximation error for CUR decomposition
        
        Args:
            original_tensor: original input tensor
            result_matrices: (C, U, R) from decomposition
            
        Returns:
            approximation error
        """
        if len(result_matrices) != 3:
            raise ValueError("Expected C, U, R")
        
        C, U, R = result_matrices
        # Reconstruct matrix: A ≈ C @ U @ R
        reconstructed = C @ U @ R
        return torch.linalg.norm(original_tensor - reconstructed)

    def _decompose(self, tensor: torch.Tensor):
        if self.stable_rank is None:
            self.stable_rank = self._get_stable_rank(tensor)
        # create sub matrices for CUR-decompostion
        c, w, r = self.select_rows_cols(tensor)
        # evaluate pseudoinverse for W - U^-1
        u = torch.linalg.pinv(w)
        # aprox U using pseudoinverse
        return (c, u, r)

    def _importance(self, X, p):
        ax = 0
        X_scaled = (X - torch.min(X, dim=ax).values) / (torch.max(X, dim=ax).values - torch.min(X, dim=ax).values)
        torch.nan_to_num_(X_scaled, 0) 
        col_norms = torch.linalg.norm(X_scaled, ord=p, axis=0)
        row_norms = torch.linalg.norm(X_scaled, ord=p, axis=1)
        matrix_norm = torch.linalg.norm(X_scaled, 'fro')  # np.sum(np.power(matrix, 2))

        # Compute the probabilities for selecting columns and rows
        col_probs, row_probs = col_norms / matrix_norm, row_norms / matrix_norm
        return col_probs, row_probs

    def select_rows_cols(
            self, X: torch.Tensor,
            p=2) -> Tuple[torch.Tensor]:
        # Evaluate norms for columns and rows
        col_probs, row_probs = self._importance(X, p)

        rank = self.stable_rank

        column_indices = torch.sort(torch.argsort(col_probs, descending=True)[:rank]).values
        row_indices = torch.sort(torch.argsort(row_probs, descending=True)[:rank]).values

        C_matrix = X[:, column_indices] 
        R_matrix = X[row_indices, :]
        W_matrix = X[row_indices, :][:, column_indices]

        return C_matrix, W_matrix, R_matrix

class RPHOSVDDecomposition(TensorDecomposer):
    """
    Random Projection Higher Order Singular Value Decomposition (RP-HOSVD).
    
    This algorithm performs Tucker decomposition using random projections for 
    computational efficiency. The method processes each tensor mode sequentially:
    
    1. For each mode n = 1, 2, ..., N:
       - Unfold tensor along mode n
       - Apply random projection to reduce dimensionality
       - Perform QR decomposition to obtain orthogonal factor matrix
    2. Compute core tensor via successive mode contractions
    
    Args:
        rank: Target rank for each mode. Can be:
            - int: Same rank for all modes
            - List[int]: Specific rank for each mode  
            - None: Auto-determined based on tensor dimensions
        distortion_factor: Random projection distortion parameter (0, 1]
        power: Power iteration parameter for improved projection quality
        random_init: Random matrix initialization ('normal' or 'ortho')
    
    Returns:
        tuple: (core_tensor, factor_matrices) representing Tucker decomposition
    """
    
    @filter_kw_universal
    def __init__(self, *, rank: Optional[Union[int, List[int]]] = None, 
                 distortion_factor: float = 0.6,
                 power: int = 3,
                 random_init: str = 'normal'):
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.rank = rank
        self.distortion_factors = distortion_factor
        self.power = power
        self.random_init = random_init
        self._random_gens = {
            'normal': lambda x, y: torch.randn(x, y),
            'ortho': _ortho_gen
        }
    
    def _decompose(self, tensor: torch.Tensor) -> tuple:
        """
        Decompose tensor using RP-HOSVD algorithm.
        
        Performs Tucker decomposition by computing factor matrices for each mode
        via random projection and QR decomposition, then contracts the tensor
        with these factors to obtain the core tensor.
        
        Args:
            tensor: Input tensor to decompose with shape (I1, I2, ..., IN)
            
        Returns:
            tuple: (core_tensor, factor_matrices) where:
                - core_tensor: Core tensor of reduced dimensions
                - factor_matrices: List of orthogonal factor matrices for each mode
        """
        # Convert rank to list if it's a single integer
        if self.rank is None:
            # Use default ranks based on tensor dimensions
            ranks = [min(dim, 10) for dim in tensor.shape]  # Default to min(dim, 10)
        elif isinstance(self.rank, int):
            ranks = [self.rank] * tensor.dim()
        else:
            ranks = self.rank
            if len(ranks) != tensor.dim():
                raise ValueError(f"Rank list length {len(ranks)} must match tensor dimensions {tensor.dim()}")
        
        self.original_shape = tensor.shape
        factor_matrices = []

        current_tensor = tensor.clone()
        for mode_idx in range(tensor.dim()):
            mode_size = current_tensor.shape[mode_idx]
            size = current_tensor.shape
            new_size = size[:mode_idx] + size[mode_idx+1:]
            ps = torch.prod(torch.tensor(new_size)).item()
            target_rank = min(ranks[mode_idx], mode_size)
            Z_n = n_mode_unfolding(current_tensor, mode_idx + 1)
            Omega = torch.randn(ps, target_rank, device=current_tensor.device, dtype=current_tensor.dtype)
            W_n = Z_n @ Omega
            Q_n, R = torch.linalg.qr(W_n)
            factor_matrices.append(Q_n)
        
        for n, matrix in enumerate(factor_matrices):
            current_tensor = self._mode_dot(current_tensor, matrix.T, n)

        return current_tensor, factor_matrices

    def _apply_random_projection(self, matrix: torch.Tensor, target_rank: int) -> torch.Tensor:
        """
        Apply random projection to reduce matrix dimensionality.
        
        Projects the input matrix to a lower-dimensional space using a random
        projection matrix. This preserves approximate distances while reducing
        computational cost for subsequent operations.
        
        Args:
            matrix: Input matrix to project with shape (m, n)
            target_rank: Target reduced dimension for projection
            
        Returns:
            torch.Tensor: Projected matrix with shape (target_rank, n)
        """
        m, n = matrix.shape
        
        # Generate random projection matrix
        if self.random_init == 'ortho':
            P = _ortho_gen(target_rank, m).to(device=matrix.device, dtype=matrix.dtype)
        else:
            P = torch.randn(target_rank, m, device=matrix.device, dtype=matrix.dtype)
        
        # Simple random projection without power iteration for now
        projected = P @ matrix
        
        return projected


class BasicRandomizedSVD(Decomposer):
    """
    Basic Randomized SVD Algorithm With Oversampling and Power Iteration
    
    This algorithm implements the randomized SVD with oversampling and power iteration
    for better approximation quality.
    
    Args:
        target_rank: target rank for decomposition
        oversampling: oversampling parameter p
        power_iteration: power iteration parameter q
        random_init: type of random initialization ('normal' or 'ortho')
    """
    
    @filter_kw_universal
    def __init__(self, *, target_rank: Optional[int] = None, oversampling: int = 10, 
                 power_iteration: int = 2, random_init: str = 'normal', distortion_factor: float = 0.6):
        self.target_rank = target_rank
        self.oversampling = oversampling
        self.power_iteration = power_iteration
        self.random_init = random_init
        self.distortion_factors = distortion_factor
        self._random_gens = {
            'normal': lambda x, y: torch.randn(x, y),
            'ortho': _ortho_gen
        }
    
    def _get_stable_rank(self, W):
        """Override _get_stable_rank for BasicRandomizedSVD"""
        n_samples = max(W.shape)
        min_num_samples = johnson_lindenstrauss_min_dim(n_samples, eps=self.distortion_factors).tolist()
        return min(round(min_num_samples), max(W.size()), 1)
    
    def _decompose(self, matrix: torch.Tensor) -> tuple:
        """
        Decompose matrix using Basic Randomized SVD
        
        Args:
            matrix: input matrix to decompose
            
        Returns:
            tuple: (U, S, V) - SVD factors
        """
        I, J = matrix.shape
        if self.target_rank is None:
            R = min(I, J) // 2  # Default to half of the minimum dimension
        else:
            R = self.target_rank
        p = self.oversampling
        q = self.power_iteration
        
        # Step 1: Generate random matrix Ω ∈ ℝ^(J × (R + p))
        if self.random_init == 'ortho':
            Omega = _ortho_gen(J, R + p).to(device=matrix.device, dtype=matrix.dtype)
        else:
            Omega = torch.randn(J, R + p, device=matrix.device, dtype=matrix.dtype)
        
        # Step 2: Form Y = (XX^T)^q XΩ
        G = matrix @ matrix.T
        # Use element-wise power like in original RandomizedSVD
        Y = torch.pow(G, q) @ matrix @ Omega
        
        # Ensure Y has the right shape
        if Y.shape[1] > R + p:
            Y = Y[:, :R + p]
        
        # Normalize Y for numerical stability
        Y = Y / torch.linalg.norm(Y, dim=0, keepdim=True)
        
        # Step 3: Compute QR decomposition Y = QR
        Q, R_qr = torch.linalg.qr(Y, mode='reduced')
        
        # Step 4: Compute B = Q^T X
        B = Q.T @ matrix
        
        # Step 5: Compute full SVD: B = ÛSṼ^T
        U_hat, S, V_hat = torch.linalg.svd(B, full_matrices=False)
        
        # Step 6: Update U = QÛ
        U = Q @ U_hat
        
        # Step 7: Truncate: U ← U(:,1:R), S ← S(1:R,1:R), V ← Ṽ(:,1:R)^T
        U = U[:, :R]
        S = S[:R]
        V = V_hat[:R, :].T
        
        return U, S, V
    
    def get_approximation_error(self, original_matrix: torch.Tensor, *result_matrices) -> torch.Tensor:
        """
        Compute approximation error for Basic Randomized SVD
        
        Args:
            original_matrix: original input matrix
            result_matrices: (U, S, V) from decomposition
            
        Returns:
            approximation error
        """
        if len(result_matrices) != 3:
            raise ValueError("Expected U, S, V")
        
        U, S, V = result_matrices
        
        # Reconstruct matrix: A ≈ USV^T
        reconstructed = U @ torch.diag(S) @ V.T
        
        return torch.linalg.norm(original_matrix - reconstructed)


class RSTHOSVDDecomposition(TensorDecomposer):
    """
    Randomized Sequentially Truncated Higher-Order SVD (R-STHOSVD).
    
    This algorithm performs Tucker decomposition by sequentially applying randomized
    SVD to tensor unfoldings and updating the core tensor. The sequential truncation
    approach reduces computational complexity while maintaining approximation quality.
    
    Algorithm:
    1. Initialize core tensor S = X
    2. For each mode n = 1, 2, ..., N:
       - Compute n-mode unfolding of current S
       - Apply randomized SVD to obtain factor matrix Q^(n)
       - Contract S with Q^(n)^T along mode n
    3. Return final core tensor and all factor matrices
    
    Args:
        rank: Target rank for each mode. Can be:
            - int: Same rank for all modes
            - List[int]: Specific rank for each mode
            - None: Auto-determined based on tensor dimensions
        oversampling: Oversampling parameter for randomized SVD stability
        power_iteration: Power iterations for improved SVD approximation
        random_init: Random matrix initialization ('normal' or 'ortho')
    
    Returns:
        tuple: (core_tensor, factor_matrices) representing Tucker decomposition
    """
    
    @filter_kw_universal
    def __init__(self, *, rank: Optional[Union[int, List[int]]] = None,
                 oversampling: int = 10,
                 power_iteration: int = 2,
                 random_init: str = 'normal'):
        self.rank = rank
        self.oversampling = oversampling
        self.power_iteration = power_iteration
        self.random_init = random_init
    
    def _decompose(self, tensor: torch.Tensor) -> tuple:
        """
        Decompose tensor using R-STHOSVD algorithm.
        
        Implements sequential truncation approach where each mode is processed
        in order, applying randomized SVD to the current tensor unfolding and
        immediately updating the core tensor. This reduces memory requirements
        compared to computing all factor matrices first.
        
        Args:
            tensor: Input tensor to decompose with shape (I1, I2, ..., IN)
            
        Returns:
            tuple: (core_tensor, factor_matrices) where:
                - core_tensor: Sequentially truncated core tensor
                - factor_matrices: List of factor matrices from randomized SVD
        """
        if tensor.dim() < 2:
            raise ValueError("Tensor must have at least 2 dimensions")
        
        # Convert rank to list if it's a single integer
        if self.rank is None:
            # Use default ranks based on tensor dimensions
            ranks = [min(dim, 10) for dim in tensor.shape]  # Default to min(dim, 10)
        elif isinstance(self.rank, int):
            ranks = [self.rank] * tensor.dim()
        else:
            ranks = self.rank
            if len(ranks) != tensor.dim():
                raise ValueError(f"Rank list length {len(ranks)} must match tensor dimensions {tensor.dim()}")
        
        # Step 1: Initialize S = X
        S = tensor.clone()
        self.original_shape = tensor.shape
        factor_matrices = []
        
        # Step 2: For each mode n = 1, 2, ..., N
        for n in range(tensor.dim()):
            # Get current mode size and target rank
            mode_size = tensor.shape[n]
            target_rank = min(ranks[n], mode_size)
            
            # Get n-unfolding of current core tensor S
            S_n = n_mode_unfolding(S, n + 1)  # n+1 because n_mode_unfolding uses 1-indexed
            
            # Apply Algorithm 1 (Basic Randomized SVD) to S^(n) with target rank R_n
            # Increase power iterations for larger tensors
            power_iter = self.power_iteration
            if tensor.numel() > 10000:  # For large tensors
                power_iter = max(self.power_iteration, 4)
            
            # Increase oversampling for larger tensors
            oversampling = self.oversampling
            if tensor.numel() > 10000:  # For large tensors
                oversampling = max(self.oversampling, 20)
            
            rsvd = BasicRandomizedSVD(
                target_rank=target_rank,
                oversampling=oversampling,
                power_iteration=power_iter,
                random_init=self.random_init
            )
            Q_n, _, _ = rsvd.decompose(S_n)
            
            # Store factor matrix Q^(n)
            factor_matrices.append(Q_n)
            
            # Update S = S ×_n Q^(n)^T
            # This contracts the tensor S with the transpose of Q_n along mode n
            # The result should be a tensor with reduced dimension along mode n
            # Note: We need to ensure that the contraction is done correctly
            # For Tucker decomposition, we need to contract along the correct mode
            S = self._mode_dot(S, Q_n.T, n)
        
        # Return core tensor S and factor matrices
        return S, factor_matrices


class RSTDecomposition(TensorDecomposer):
    """
    Randomized Sampling Tucker (R-ST) Decomposition.
    
    This algorithm performs Tucker decomposition using probabilistic column sampling
    from tensor mode unfoldings. Instead of computing full SVDs, it selects the most
    important columns based on various sampling strategies, making it suitable for
    very large tensors where memory is constrained.
    
    Algorithm:
    1. For each mode n = 1, 2, ..., N:
       - Compute n-mode unfolding X(n)
       - Sample columns based on importance (norm, leverage scores, etc.)
       - Store sampled columns as factor matrix Q(n)
    2. Compute core tensor via successive pseudoinverse contractions:
       S = X ×₁ Q₁^† ×₂ Q₂^† ... ×ₙ Qₙ^†
    
    Args:
        rank: Target rank for each mode. Can be:
            - int: Same rank for all modes
            - List[int]: Specific rank for each mode
            - None: Auto-determined based on tensor dimensions
        sampling_method: Column sampling strategy:
            - 'uniform': Random uniform sampling
            - 'norm_based': Probability proportional to column norms
            - 'leverage_score': Sampling based on statistical leverage
        distortion_factor: Controls approximation quality vs speed trade-off
        random_init: Random matrix initialization ('normal' or 'ortho')
    
    Returns:
        tuple: (core_tensor, factor_matrices) representing Tucker decomposition
    """
    
    @filter_kw_universal
    def __init__(self, *, rank: Optional[Union[int, List[int]]] = None,
                 sampling_method: str = 'norm_based',
                 distortion_factor: float = 0.6,
                 random_init: str = 'normal'):
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.rank = rank
        self.sampling_method = sampling_method
        self.distortion_factors = distortion_factor
        self.random_init = random_init
        self._random_gens = {
            'normal': lambda x, y: torch.randn(x, y),
            'ortho': _ortho_gen
        }
    
    def _decompose(self, tensor: torch.Tensor) -> tuple:
        """
        Decompose tensor using R-ST algorithm.
        
        Performs Tucker decomposition by sampling important columns from each
        tensor mode unfolding based on the specified sampling strategy, then
        computing the core tensor via pseudoinverse contractions.
        
        Args:
            tensor: Input tensor to decompose with shape (I1, I2, ..., IN)
            
        Returns:
            tuple: (core_tensor, factor_matrices) where:
                - core_tensor: Core tensor computed via pseudoinverse contractions
                - factor_matrices: List of sampled column matrices for each mode
        """
        if tensor.dim() < 2:
            raise ValueError("Tensor must have at least 2 dimensions")
        
        # Convert rank to list if it's a single integer
        if self.rank is None:
            # Use default ranks based on tensor dimensions
            ranks = [min(dim, 10) for dim in tensor.shape]  # Default to min(dim, 10)
        elif isinstance(self.rank, int):
            ranks = [self.rank] * tensor.dim()
        else:
            ranks = self.rank
            if len(ranks) != tensor.dim():
                raise ValueError(f"Rank list length {len(ranks)} must match tensor dimensions {tensor.dim()}")
        
        # Store original tensor shape
        self.original_shape = tensor.shape
        factor_matrices = []
        
        # Step 1: For each mode n = 1, 2, ..., N
        for mode_idx in range(tensor.dim()):
            mode_size = tensor.shape[mode_idx]
            target_rank = min(ranks[mode_idx], mode_size)

            # Get n-mode unfolding efficiently
            unfolding = n_mode_unfolding(tensor, mode_idx + 1)
            
            # Sample columns using the specified method
            Q_n = self._sample_columns(unfolding, target_rank)
            factor_matrices.append(Q_n)
        
        # Step 2: Compute core tensor efficiently
        core_tensor = self._compute_core_tensor(tensor, factor_matrices)
        
        return core_tensor, factor_matrices
    
    def _sample_columns(self, matrix: torch.Tensor, target_rank: int) -> torch.Tensor:
        """
        Sample columns from matrix based on importance sampling strategy.
        
        Selects the most important columns from the tensor mode unfolding
        according to the specified sampling method. This reduces computational
        cost compared to full SVD while preserving the most significant information.
        
        Args:
            matrix: Tensor mode unfolding with shape (mode_size, other_dims_product)
            target_rank: Number of columns to sample for this mode
            
        Returns:
            torch.Tensor: Sampled columns matrix with shape (mode_size, target_rank)
        """
        In, J = matrix.shape  # In is mode size, J is product of other dimensions
        
        if self.sampling_method == 'uniform':
            # Uniform random sampling
            indices = torch.randperm(J, device=matrix.device)[:target_rank]
            Q_n = matrix[:, indices]
            
        elif self.sampling_method == 'norm_based':
            # Sample based on column norms (importance sampling)
            col_norms = torch.linalg.norm(matrix, dim=0)
            # Normalize to get probabilities
            probs = col_norms / torch.sum(col_norms)
            # Sample with replacement based on probabilities
            indices = torch.multinomial(probs, target_rank, replacement=False)
            Q_n = matrix[:, indices]
            
        elif self.sampling_method == 'leverage_score':
            # Sample based on leverage scores
            # Compute leverage scores using SVD
            U, S, V = torch.linalg.svd(matrix, full_matrices=False)
            # Leverage scores are squared row norms of U
            leverage_scores = torch.sum(U**2, dim=1)
            # Normalize to get probabilities
            probs = leverage_scores / torch.sum(leverage_scores)
            # Sample with replacement based on probabilities
            indices = torch.multinomial(probs, target_rank, replacement=False)
            Q_n = matrix[:, indices]
            
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        
        # Ensure Q_n has the correct shape
        if Q_n.shape[1] > target_rank:
            Q_n = Q_n[:, :target_rank]
        elif Q_n.shape[1] < target_rank:
            # Pad with zeros if we don't have enough columns
            padding = torch.zeros(In, target_rank - Q_n.shape[1], dtype=Q_n.dtype, device=Q_n.device)
            Q_n = torch.cat([Q_n, padding], dim=1)
        
        return Q_n
    
    def _compute_core_tensor(self, tensor: torch.Tensor, factor_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute core tensor via successive pseudoinverse contractions.
        
        Contracts the original tensor with the pseudoinverse of each factor matrix
        along the corresponding mode: S = X ×₁ Q₁^† ×₂ Q₂^† ... ×ₙ Qₙ^†
        This gives the optimal core tensor in the least-squares sense for the
        given factor matrices.
        
        Args:
            tensor: Original input tensor with shape (I1, I2, ..., IN)
            factor_matrices: List of sampled factor matrices [Q₁, Q₂, ..., Qₙ]
            
        Returns:
            torch.Tensor: Core tensor with reduced dimensions matching factor matrix ranks
        """
        # Start with the original tensor
        core_tensor = tensor.clone()
        
        # For each mode, contract with the pseudoinverse of the factor matrix
        for mode_idx, Q_n in enumerate(factor_matrices):
            # Compute pseudoinverse of Q_n
            Q_n_pinv = torch.linalg.pinv(Q_n)
            
            # Use mode_dot for proper tensor contraction
            core_tensor = self._mode_dot(core_tensor, Q_n_pinv, mode_idx)
        
        return core_tensor


DECOMPOSERS = {
    'svd': SVDDecomposition,
    'rsvd': RandomizedSVD,
    'cur': CURDecomposition,
    'rphosvd': RPHOSVDDecomposition,
    'basic_rsvd': BasicRandomizedSVD,
    'rsthosvd': RSTHOSVDDecomposition,
    'rst': RSTDecomposition,
    }
