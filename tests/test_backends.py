from __future__ import annotations

import numpy as np
from conftest import make_alternating_tensor, skip_no_cupy, skip_no_jax

from tensorrsvd import ho_rsvd

K, SHAPE, RANK = 3, (6, 6, 6), 3


def _smoke(backend: str):
    fn = make_alternating_tensor(K)
    U_list, S_list = ho_rsvd(
        tensor=fn,
        tensor_shape=SHAPE,
        dtype=np.float64,
        rank=RANK,
        num_oversamples=5,
        num_idxs=K,
        backend=backend,
    )
    assert len(U_list) == K
    assert len(S_list) == K
    for i in range(K):
        assert U_list[i].shape == (SHAPE[i], RANK)
        assert S_list[i].shape == (RANK,)
    return U_list, S_list


def test_numpy_smoke():
    _smoke("numpy")


def test_numpy_output_types():
    U_list, S_list = _smoke("numpy")
    for U in U_list:
        assert isinstance(U, np.ndarray)
    for S in S_list:
        assert isinstance(S, np.ndarray)


@skip_no_jax
def test_jax_smoke():
    _smoke("jax")


@skip_no_jax
def test_jax_output_types():
    import jax.numpy as jnp  # noqa: F401

    U_list, S_list = _smoke("jax")
    for U in U_list:
        assert hasattr(U, "shape"), "JAX array should have .shape"
    for S in S_list:
        assert hasattr(S, "shape"), "JAX array should have .shape"


@skip_no_jax
def test_jax_orthonormality():
    import jax.numpy as jnp

    U_list, _ = _smoke("jax")
    for i, U in enumerate(U_list):
        UtU = jnp.array(U).T @ jnp.array(U)
        np.testing.assert_allclose(
            np.array(UtU),
            np.eye(RANK),
            atol=1e-8,
            err_msg=f"JAX U_list[{i}] is not orthonormal",
        )


@skip_no_cupy
def test_cupy_smoke():
    _smoke("cupy")


@skip_no_cupy
def test_cupy_output_types():
    import cupy as cp

    U_list, S_list = _smoke("cupy")
    for U in U_list:
        assert isinstance(U, cp.ndarray)
    for S in S_list:
        assert isinstance(S, cp.ndarray)
