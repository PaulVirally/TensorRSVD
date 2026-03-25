# TensorRSVD

[![PyPI version](https://img.shields.io/pypi/v/tensorrsvd.svg)](https://pypi.org/project/tensorrsvd/)
[![CI](https://github.com/PaulVirally/TensorRSVD/actions/workflows/ci.yml/badge.svg)](https://github.com/PaulVirally/TensorRSVD/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/tensorrsvd/badge/?version=latest)](https://tensorrsvd.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Tensor-free randomized Higher-Order SVD (Tucker decomposition).**

TensorRSVD computes Tucker decompositions of high-dimensional tensors that are
defined as Python functions without ever forming the dense tensor in memory. It
uses randomized linear algebra (Halko et al. 2011) to approximate the dominant
factor matrices mode-by-mode, scaling to tensors that would be impossible to
store explicitly.

## Installation

```bash
pip install tensorrsvd
```

**Optional backends** for GPU or JIT-compiled acceleration:

```bash
pip install "tensorrsvd[jax]"          # JAX (CPU / GPU / TPU)
pip install "tensorrsvd[jaxcuda12]"    # JAX with CUDA 12
pip install "tensorrsvd[jaxcuda13]"    # JAX with CUDA 13
```

CuPy (NVIDIA GPU) must be installed separately following the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

## Quick start

```python
import numpy as np
from tensorrsvd import ho_svd_r

# Define your tensor as a callable — no dense array needed
def my_tensor(x0, x1, x2):
    return x0 - x1 + x2

# Compute the randomized Tucker factors
U_list, S_list = ho_svd_r(
    tensor=my_tensor,
    tensor_shape=(64, 64, 64),
    dtype=np.float64,
    rank=4,
    num_oversamples=10,
    num_idxs=3,
)

# U_list[i] has shape (n_i, rank_i) — orthonormal columns
# S_list[i] has shape (rank_i,)      — singular values, descending
print([U.shape for U in U_list])   # [(64, 4), (64, 4), (64, 4)]
```

Switch to a JAX or CuPy backend by passing `backend="jax"` or `backend="cupy"`.

## Documentation

Full documentation (installation, user guide, theory, and API reference, etc.)
is available at
**[tensorrsvd.readthedocs.io](https://tensorrsvd.readthedocs.io)**.

## References

- N. Halko, P. G. Martinsson, and J. A. Tropp,
  *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions*,
  SIAM Review, 53(2):217–288, 2011.
  [doi:10.1137/090771806](https://doi.org/10.1137/090771806)

- L. De Lathauwer, B. De Moor, and J. Vandewalle,
  *A Multilinear Singular Value Decomposition*,
  SIAM Journal on Matrix Analysis and Applications, 21(4):1253–1278, 2000.
  [doi:10.1137/S0895479896305696](https://doi.org/10.1137/S0895479896305696)

## License

MIT (see [LICENSE](LICENSE))
