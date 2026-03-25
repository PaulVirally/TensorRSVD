from __future__ import annotations

import numpy as np
import pytest
from conftest import make_alternating_tensor, numpy_dense_tensor

from tensorrsvd.core import MatricizedTensorOperator


def dense_mode_unfolding(T: np.ndarray, mode: int) -> np.ndarray:
    """Mode-m unfolding of a dense tensor T.

    Returns a 2-D array of shape (T.shape[mode], prod(other dims)) using the
    column ordering produced by moving mode to the front and ravelling the rest.
    """
    ndim = T.ndim
    order = [mode] + [i for i in range(ndim) if i != mode]
    return T.transpose(order).reshape(T.shape[mode], -1)


@pytest.mark.parametrize(
    "shape,mode,expected_op_shape",
    [
        ((4, 5), 0, (4, 5)),
        ((4, 5), 1, (5, 4)),
        ((3, 4, 5), 0, (3, 20)),
        ((3, 4, 5), 1, (4, 15)),
        ((3, 4, 5), 2, (5, 12)),
        ((3, 3, 3, 3), 0, (3, 27)),
        ((3, 3, 3, 3), 2, (3, 27)),
    ],
)
def test_operator_shape(shape, mode, expected_op_shape):
    k = len(shape)
    fn = make_alternating_tensor(k)
    op = MatricizedTensorOperator(fn, shape, mode, np.float64, "numpy")
    assert op.shape == expected_op_shape


@pytest.mark.parametrize(
    "shape,mode",
    [
        ((4, 5), 0),
        ((4, 5), 1),
        ((3, 4, 5), 1),
    ],
)
def test_eval_row_matches_dense_unfolding(shape, mode):
    k = len(shape)
    fn = make_alternating_tensor(k)
    T_dense = numpy_dense_tensor(fn, shape)
    A_dense = dense_mode_unfolding(T_dense, mode)

    op = MatricizedTensorOperator(fn, shape, mode, np.float64, "numpy")
    for i in range(shape[mode]):
        row = op._eval_row(i)
        np.testing.assert_allclose(row, A_dense[i], atol=1e-12)


@pytest.mark.parametrize(
    "shape,mode",
    [
        ((4, 5), 0),
        ((4, 5), 1),
        ((3, 4, 5), 0),
        ((3, 4, 5), 2),
        ((3, 3, 3, 3), 1),
    ],
)
def test_matmat_matches_dense(shape, mode):
    rng = np.random.default_rng(42)
    k = len(shape)
    fn = make_alternating_tensor(k)
    T_dense = numpy_dense_tensor(fn, shape)
    A_dense = dense_mode_unfolding(T_dense, mode)

    op = MatricizedTensorOperator(fn, shape, mode, np.float64, "numpy")
    num_cols = A_dense.shape[1]
    out_cols = 4
    X = rng.standard_normal((num_cols, out_cols))
    np.testing.assert_allclose(op._matmat(X), A_dense @ X, atol=1e-10)


@pytest.mark.parametrize(
    "shape,mode",
    [
        ((4, 5), 0),
        ((5, 6), 1),
        ((3, 4, 5), 2),
    ],
)
def test_matvec_matches_dense(shape, mode):
    rng = np.random.default_rng(7)
    k = len(shape)
    fn = make_alternating_tensor(k)
    T_dense = numpy_dense_tensor(fn, shape)
    A_dense = dense_mode_unfolding(T_dense, mode)

    op = MatricizedTensorOperator(fn, shape, mode, np.float64, "numpy")
    x = rng.standard_normal(A_dense.shape[1])
    np.testing.assert_allclose(op._matvec(x), A_dense @ x, atol=1e-10)


@pytest.mark.parametrize(
    "shape,mode",
    [
        ((4, 5), 0),
        ((4, 5), 1),
        ((3, 4, 5), 1),
        ((3, 3, 3, 3), 2),
    ],
)
def test_adjoint_consistency(shape, mode):
    rng = np.random.default_rng(99)
    k = len(shape)
    fn = make_alternating_tensor(k)
    op = MatricizedTensorOperator(fn, shape, mode, np.float64, "numpy")

    m, n = op.shape
    x = rng.standard_normal(n)
    y = rng.standard_normal(m)

    lhs = np.dot(op._matvec(x), y)
    rhs = np.dot(x, op._rmatvec(y))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


@pytest.mark.parametrize(
    "shape,mode",
    [
        ((4, 5), 0),
        ((3, 4, 5), 0),
    ],
)
def test_matmat_adjoint_consistency(shape, mode):
    rng = np.random.default_rng(13)
    k = len(shape)
    fn = make_alternating_tensor(k)
    op = MatricizedTensorOperator(fn, shape, mode, np.float64, "numpy")

    m, n = op.shape
    out_cols = 3
    X = rng.standard_normal((n, out_cols))
    Y = rng.standard_normal((m, out_cols))

    lhs = np.sum(op._matmat(X) * Y)
    rhs = np.sum(X * op._rmatmat(Y))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)
