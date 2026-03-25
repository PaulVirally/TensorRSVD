from __future__ import annotations

import numpy as np
import pylops
import pytest
from conftest import make_alternating_tensor, numpy_dense_tensor

from tensorrsvd.core import MatricizedTensorOperator, randomized_range_finder, rsvd_left


def dense_mode_unfolding(T: np.ndarray, mode: int) -> np.ndarray:
    ndim = T.ndim
    order = [mode] + [i for i in range(ndim) if i != mode]
    return T.transpose(order).reshape(T.shape[mode], -1)


class TestRandomizedRangeFinder:
    def _make_op(self, shape=(8, 8), mode=0, k=2):
        fn = make_alternating_tensor(k)
        return MatricizedTensorOperator(fn, shape, mode, np.float64, "numpy")

    def test_output_is_orthonormal(self):
        op = self._make_op()
        rank, oversamples = 4, 4
        Q = randomized_range_finder(op, rank, oversamples, 0, "numpy")
        assert Q.shape == (op.shape[0], rank + oversamples)
        QtQ = Q.T @ Q
        np.testing.assert_allclose(QtQ, np.eye(rank + oversamples), atol=1e-10)

    def test_range_approximation_quality(self):
        """|| A - Q Q^T A ||_F / || A ||_F should be small for full rank."""
        fn = make_alternating_tensor(3)
        shape = (10, 10, 10)
        T = numpy_dense_tensor(fn, shape)
        A = dense_mode_unfolding(T, 0)

        op = MatricizedTensorOperator(fn, shape, 0, np.float64, "numpy")
        rank, oversamples = 3, 5
        Q = randomized_range_finder(op, rank, oversamples, 0, "numpy")

        residual = A - Q @ (Q.T @ A)
        rel_err = np.linalg.norm(residual, "fro") / np.linalg.norm(A, "fro")
        assert rel_err < 0.01, f"Range approximation error too large: {rel_err}"

    def test_power_iterations_improve_quality(self):
        """More power iterations should not worsen the approximation."""
        fn = make_alternating_tensor(3)
        shape = (8, 8, 8)
        T = numpy_dense_tensor(fn, shape)
        A = dense_mode_unfolding(T, 0)
        op = MatricizedTensorOperator(fn, shape, 0, np.float64, "numpy")

        rank, oversamples = 2, 4

        errors = []
        for p in [0, 1, 2]:
            Q = randomized_range_finder(op, rank, oversamples, p, "numpy")
            residual = A - Q @ (Q.T @ A)
            errors.append(np.linalg.norm(residual, "fro") / np.linalg.norm(A, "fro"))

        # Each additional power iteration should not increase the error
        assert errors[1] <= errors[0] + 1e-6
        assert errors[2] <= errors[1] + 1e-6

    @pytest.mark.parametrize("oversamples", [0, 5, 10])
    def test_orthonormal_with_various_oversamples(self, oversamples):
        op = self._make_op(shape=(4, 30), mode=0, k=2)
        rank = 2
        Q = randomized_range_finder(op, rank, oversamples, 0, "numpy")
        QtQ = Q.T @ Q
        # Q has shape (m, min(rank+oversamples, m)) due to QR clamping
        np.testing.assert_allclose(QtQ, np.eye(Q.shape[1]), atol=1e-10)


class TestRsvdLeft:
    def test_singular_values_match_numpy_svd(self):
        """Randomized singular values should be close to exact SVD values."""
        fn = make_alternating_tensor(3)
        shape = (10, 10, 10)
        T = numpy_dense_tensor(fn, shape)
        A = dense_mode_unfolding(T, 0)

        op = MatricizedTensorOperator(fn, shape, 0, np.float64, "numpy")
        rank = 3
        U_r, S_r = rsvd_left(op, rank, num_oversamples=10, num_power_iterations=2, backend="numpy")

        _, S_exact, _ = np.linalg.svd(A, full_matrices=False)
        S_exact = S_exact[:rank]

        np.testing.assert_allclose(
            S_r,
            S_exact,
            rtol=0.05,
            atol=1e-10,
            err_msg="Randomized singular values deviate from exact",
        )

    def test_left_singular_vectors_orthonormal(self):
        fn = make_alternating_tensor(3)
        shape = (8, 8, 8)
        op = MatricizedTensorOperator(fn, shape, 1, np.float64, "numpy")
        rank = 3
        U, _ = rsvd_left(op, rank, num_oversamples=5, num_power_iterations=1, backend="numpy")

        assert U.shape == (op.shape[0], rank)
        UtU = U.T @ U
        np.testing.assert_allclose(UtU, np.eye(rank), atol=1e-10)

    def test_singular_values_non_negative(self):
        fn = make_alternating_tensor(2)
        shape = (6, 8)
        op = MatricizedTensorOperator(fn, shape, 0, np.float64, "numpy")
        _, S = rsvd_left(op, rank=2, num_oversamples=4, num_power_iterations=0, backend="numpy")
        assert np.all(S >= 0)

    @pytest.mark.parametrize(
        "shape,mode",
        [
            ((5, 7), 0),
            ((5, 7), 1),
            ((4, 5, 6), 0),
            ((4, 5, 6), 2),
        ],
    )
    def test_output_shapes(self, shape, mode):
        k = len(shape)
        fn = make_alternating_tensor(k)
        op = MatricizedTensorOperator(fn, shape, mode, np.float64, "numpy")
        rank = min(2, op.shape[0], op.shape[1])
        U, S = rsvd_left(op, rank, num_oversamples=3, num_power_iterations=0, backend="numpy")
        assert U.shape == (op.shape[0], rank)
        assert S.shape == (rank,)


def _subspace_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """Frobenius norm of the difference between two projection matrices."""
    return np.linalg.norm(U1 @ U1.T - U2 @ U2.T, "fro")


class TestRsvdLeftDenseMatrices:
    """Test rsvd_left on hand-crafted dense matrices via pylops.MatrixMult.

    Three matrix classes are covered:
      1. Diagonal
      2. Symmetric
      3. Generic rectangular
    """

    # ------------------------------------------------------------------
    # Matrix 1: diagonal (6 x 10)
    #   singular values: 20, 10, 5, 2, 1, 0.5  (in decreasing order)
    #   left singular vectors: first 6 standard basis vectors of R^6
    # ------------------------------------------------------------------
    DIAG_VALS = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    DIAG_MAT = np.zeros((6, 10), dtype=np.float64)
    for _i, _v in enumerate(DIAG_VALS):
        DIAG_MAT[_i, _i] = _v

    # ------------------------------------------------------------------
    # Matrix 2: symmetric positive definite (8 x 8) tridiagonal-like
    #   H_{ii} = 6, H_{i,i±1} = 2, H_{i,i±2} = 1  (within bounds)
    # ------------------------------------------------------------------
    _H = np.zeros((8, 8), dtype=np.float64)
    for _i in range(8):
        _H[_i, _i] = 6.0
        if _i + 1 < 8:
            _H[_i, _i + 1] = _H[_i + 1, _i] = 2.0
        if _i + 2 < 8:
            _H[_i, _i + 2] = _H[_i + 2, _i] = 1.0
    SYMM_MAT = _H

    # ------------------------------------------------------------------
    # Matrix 3: generic rectangular (9 x 6) – no special structure
    # ------------------------------------------------------------------
    GENERIC_MAT = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 17, 18, 19],
            [1, 3, 5, 7, 11, 13],
            [2, 4, 6, 8, 10, 14],
            [17, 3, 8, 11, 5, 9],
            [1, 7, 4, 12, 6, 2],
            [10, 6, 3, 8, 4, 15],
            [5, 9, 2, 14, 7, 3],
        ],
        dtype=np.float64,
    )

    @staticmethod
    def _op(A: np.ndarray) -> pylops.LinearOperator:
        return pylops.MatrixMult(A, dtype=A.dtype)

    @staticmethod
    def _exact_svd(A: np.ndarray, rank: int):
        U_ex, S_ex, _ = np.linalg.svd(A, full_matrices=False)
        return U_ex[:, :rank], S_ex[:rank]

    def test_diagonal_singular_values(self):
        rank = 5  # one less than true rank of 6 to leave room for oversampling
        op = self._op(self.DIAG_MAT)
        _, S_ex = self._exact_svd(self.DIAG_MAT, rank)

        _, S_r = rsvd_left(op, rank, num_oversamples=4, num_power_iterations=2, backend="numpy")

        np.testing.assert_allclose(
            S_r, S_ex, rtol=1e-6, atol=1e-10, err_msg="Diagonal matrix: singular values mismatch"
        )

    def test_diagonal_left_singular_vectors(self):
        rank = 5
        op = self._op(self.DIAG_MAT)
        U_ex, _ = self._exact_svd(self.DIAG_MAT, rank)

        U_r, _ = rsvd_left(op, rank, num_oversamples=4, num_power_iterations=2, backend="numpy")

        assert U_r.shape == (6, rank)
        assert _subspace_distance(U_r, U_ex) < 1e-8, (
            f"Diagonal matrix: left singular subspace error too large: "
            f"{_subspace_distance(U_r, U_ex):.2e}"
        )

    def test_symmetric_singular_values(self):
        # Symmetric PD: singular values == eigenvalues, all positive.
        # Use rank = 6 out of 8; n = 8, oversamples = 2 → l = 8 = n (valid).
        rank = 6
        op = self._op(self.SYMM_MAT)
        _, S_ex = self._exact_svd(self.SYMM_MAT, rank)

        _, S_r = rsvd_left(op, rank, num_oversamples=2, num_power_iterations=2, backend="numpy")

        np.testing.assert_allclose(
            S_r, S_ex, rtol=1e-6, atol=1e-10, err_msg="Symmetric matrix: singular values mismatch"
        )

    def test_symmetric_left_singular_vectors(self):
        rank = 6
        op = self._op(self.SYMM_MAT)
        U_ex, _ = self._exact_svd(self.SYMM_MAT, rank)

        U_r, _ = rsvd_left(op, rank, num_oversamples=2, num_power_iterations=2, backend="numpy")

        assert U_r.shape == (8, rank)
        assert _subspace_distance(U_r, U_ex) < 1e-8, (
            f"Symmetric matrix: left singular subspace error too large: "
            f"{_subspace_distance(U_r, U_ex):.2e}"
        )

    def test_generic_singular_values(self):
        # Shape 9x6; true rank = 6. rank=4, oversamples=2 l=6=n (valid).
        rank = 4
        op = self._op(self.GENERIC_MAT)
        _, S_ex = self._exact_svd(self.GENERIC_MAT, rank)

        _, S_r = rsvd_left(op, rank, num_oversamples=2, num_power_iterations=2, backend="numpy")

        np.testing.assert_allclose(
            S_r, S_ex, rtol=1e-6, atol=1e-10, err_msg="Generic matrix: singular values mismatch"
        )

    def test_generic_left_singular_vectors(self):
        rank = 4
        op = self._op(self.GENERIC_MAT)
        U_ex, _ = self._exact_svd(self.GENERIC_MAT, rank)

        U_r, _ = rsvd_left(op, rank, num_oversamples=2, num_power_iterations=2, backend="numpy")

        assert U_r.shape == (9, rank)
        assert _subspace_distance(U_r, U_ex) < 1e-8, (
            f"Generic matrix: left singular subspace error too large: "
            f"{_subspace_distance(U_r, U_ex):.2e}"
        )
