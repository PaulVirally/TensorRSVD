Installation
============

Requirements
------------

- **Python** ≥ 3.14
- **NumPy** ≥ 2.4
- **pylops** ≥ 2.6

These are installed automatically when you install TensorRSVD from PyPI.

From PyPI
---------

.. code-block:: bash

   pip install tensorrsvd

Optional backends
-----------------

TensorRSVD ships with optional support for two accelerated backends.

**JAX** (CPU / GPU / TPU)

JAX enables JIT-compiled matrix–vector products inside
:class:`~tensorrsvd.core.MatricizedTensorOperator` and is required for the
``backend="jax"`` option.

.. code-block:: bash

   pip install "tensorrsvd[jax]"

For CUDA 12 or CUDA 13 support:

.. code-block:: bash

   pip install "tensorrsvd[jaxcuda12]"   # CUDA 12
   pip install "tensorrsvd[jaxcuda13]"   # CUDA 13

See the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_
for additional platform-specific instructions.

**CuPy** (NVIDIA GPU only)

CuPy offloads all linear-algebra primitives to a CUDA device and is required
for the ``backend="cupy"`` option. Install it following the
`CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_,
choosing the wheel that matches your CUDA version.

Verifying the installation
--------------------------

Run a quick smoke test from a Python shell:

.. code-block:: python

   import numpy as np
   from tensorrsvd import ho_svd_r

   def my_tensor(x0, x1, x2):
       return x0 - x1 + x2

   U_list, S_list = ho_svd_r(
       tensor=my_tensor,
       tensor_shape=(16, 16, 16),
       dtype=np.float64,
       rank=3,
       num_oversamples=5,
       num_idxs=3,
   )
   print([U.shape for U in U_list])  # [(16, 3), (16, 3), (16, 3)]

If no exception is raised and the shapes look right, the installation is
working correctly.
