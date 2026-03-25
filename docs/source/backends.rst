Backends
========

TensorRSVD supports three array backends: **NumPy** (default), **JAX**, and
**CuPy**. The backend is selected by passing the ``backend`` keyword argument to
any top-level or core function.

If your tensor is defined through JAX, the callable (the tensor function) must
be JAX-traceable. This means it must be compatible with JAX's JIT compilation
(i.e., it must be a pure function without side effects) and it must use JAX's
array operations only.

All linear-algebra primitives (QR, SVD, random normal sampling) and array
creation routines are resolved at runtime through the helper functions below,
which return the correct callable for the requested backend.

.. currentmodule:: tensorrsvd.backends

.. rubric:: Linear algebra

.. autofunction:: get_qr
.. autofunction:: get_svd

.. rubric:: Random sampling

.. autofunction:: get_normal

.. rubric:: Array creation

.. autofunction:: get_arange
.. autofunction:: get_meshgrid
.. autofunction:: get_empty
.. autofunction:: get_zeros

.. rubric:: Element-wise operations

.. autofunction:: get_conj
.. autofunction:: get_ravel

.. rubric:: Dtype utilities

.. autofunction:: is_complex
.. autofunction:: real_dtype
