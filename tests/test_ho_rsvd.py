from __future__ import annotations

import numpy as np
import pytest
from conftest import make_alternating_tensor

from tensorrsvd import ho_rsvd, reconstruct


@pytest.mark.parametrize(
    "k,shape,rank",
    [
        (2, (8, 8), 2),
        (3, (8, 8, 8), 3),
        (4, (6, 6, 6, 6), 4),
        (5, (5, 5, 5, 5, 5), 5),
    ],
)
def test_smoke_runs_without_error(k, shape, rank):
    fn = make_alternating_tensor(k)
    U_list, S_list = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=rank,
        num_oversamples=5,
        num_idxs=k,
        backend="numpy",
    )
    assert len(U_list) == k
    assert len(S_list) == k
    for i in range(k):
        assert U_list[i].shape == (shape[i], rank)
        assert S_list[i].shape == (rank,)


def test_num_idxs_inferred_from_rank_list():
    k = 3
    shape = (6, 6, 6)
    fn = make_alternating_tensor(k)
    U_list, S_list = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=[3, 3, 3],
        num_oversamples=4,
        backend="numpy",
    )
    assert len(U_list) == k
    assert len(S_list) == k


def test_num_idxs_inferred_from_oversamples_list():
    k = 3
    shape = (6, 6, 6)
    fn = make_alternating_tensor(k)
    U_list, S_list = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=3,
        num_oversamples=[4, 4, 4],
        backend="numpy",
    )
    assert len(U_list) == k


def test_explicit_num_idxs_with_scalar_rank():
    k = 3
    shape = (6, 6, 6)
    fn = make_alternating_tensor(k)
    U_list, S_list = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=3,
        num_idxs=k,
        backend="numpy",
    )
    assert len(U_list) == k


@pytest.mark.parametrize(
    "k,shape,rank",
    [
        (2, (8, 8), 2),
        (3, (8, 8, 8), 3),
        (4, (6, 6, 6, 6), 4),
    ],
)
def test_factor_matrices_are_orthonormal(k, shape, rank):
    fn = make_alternating_tensor(k)
    U_list, _ = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=rank,
        num_oversamples=5,
        num_idxs=k,
        backend="numpy",
    )
    for i, U in enumerate(U_list):
        UtU = U.T @ U
        np.testing.assert_allclose(
            UtU,
            np.eye(rank),
            atol=1e-10,
            err_msg=f"U_list[{i}] is not orthonormal",
        )


@pytest.mark.parametrize(
    "k,shape,rank",
    [
        (2, (8, 8), 2),
        (3, (8, 8, 8), 3),
    ],
)
def test_singular_values_non_negative_and_decreasing(k, shape, rank):
    fn = make_alternating_tensor(k)
    _, S_list = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=rank,
        num_oversamples=5,
        num_idxs=k,
        backend="numpy",
    )
    for i, S in enumerate(S_list):
        assert np.all(S >= 0), f"S_list[{i}] has negative values"
        diffs = np.diff(S)
        assert np.all(diffs <= 1e-10), f"S_list[{i}] is not non-increasing: {S}"


def test_invalid_backend_raises():
    fn = make_alternating_tensor(2)
    with pytest.raises(ValueError, match="Unsupported backend"):
        ho_rsvd(fn, (4, 4), np.float64, rank=2, num_idxs=2, backend="bogus")


def test_all_scalar_without_num_idxs_raises():
    fn = make_alternating_tensor(2)
    with pytest.raises(ValueError):
        ho_rsvd(
            fn,
            (4, 4),
            np.float64,
            rank=2,
            num_oversamples=5,
            num_power_iterations=0,
            num_idxs=None,
            backend="numpy",
        )


def test_rank_list_length_mismatch_raises():
    fn = make_alternating_tensor(3)
    with pytest.raises(ValueError):
        ho_rsvd(fn, (4, 4, 4), np.float64, rank=[2, 2], num_idxs=3, backend="numpy")


@pytest.mark.parametrize(
    "k,shape,rank",
    [
        (2, (8, 8), 2),
        (3, (8, 8, 8), 3),
    ],
)
def test_reconstruct_output_shape(k, shape, rank):
    fn = make_alternating_tensor(k)
    U_list, _ = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=rank,
        num_oversamples=5,
        num_idxs=k,
        backend="numpy",
    )
    T_hat = reconstruct(fn, shape, U_list, dtype=np.float64)
    assert T_hat.shape == shape
    assert T_hat.dtype == np.float64


@pytest.mark.parametrize(
    "k,shape,rank",
    [
        (2, (8, 8), 2),
        (3, (8, 8, 8), 3),
    ],
)
def test_reconstruct_exact_for_rank_k_tensor(k, shape, rank):
    """Alternating tensor has exact Tucker rank k, so rank-k reconstruction
    should recover the original up to floating-point noise."""
    fn = make_alternating_tensor(k)
    U_list, _ = ho_rsvd(
        tensor=fn,
        tensor_shape=shape,
        dtype=np.float64,
        rank=rank,
        num_oversamples=5,
        num_idxs=k,
        backend="numpy",
    )
    T_hat = reconstruct(fn, shape, U_list, dtype=np.float64)

    grids = [np.arange(n) / max(n - 1, 1) for n in shape]
    coords = np.meshgrid(*grids, indexing="ij")
    T_true = fn(*coords)

    rel_err = np.linalg.norm(T_true - T_hat) / np.linalg.norm(T_true)
    assert rel_err < 1e-8, f"Relative error {rel_err:.2e} too large for exact-rank tensor"


def test_reconstruct_error_decreases_with_rank():
    """Higher rank should give strictly lower reconstruction error for a
    tensor with Tucker rank > 1 in every mode."""
    k = 3
    shape = (16, 16, 16)

    # x0*x1 + x1*x2 + x0*x2 is not rank-1 in any mode, so rank-1 gives a
    # non-trivial error while rank-3 gives near-zero error.
    def pairwise(x0, x1, x2):
        return x0 * x1 + x1 * x2 + x0 * x2

    grids = [np.arange(n) / max(n - 1, 1) for n in shape]
    coords = np.meshgrid(*grids, indexing="ij")
    T_true = pairwise(*coords)
    norm_T = np.linalg.norm(T_true)

    errors = []
    for rank in (1, 3):
        U_list, _ = ho_rsvd(
            tensor=pairwise,
            tensor_shape=shape,
            dtype=np.float64,
            rank=rank,
            num_oversamples=5,
            num_idxs=k,
            backend="numpy",
        )
        T_hat = reconstruct(pairwise, shape, U_list, dtype=np.float64)
        errors.append(float(np.linalg.norm(T_true - T_hat) / norm_T))

    assert errors[1] < errors[0], (
        f"Error at rank 3 ({errors[1]:.4f}) should be < error at rank 1 ({errors[0]:.4f})"
    )
