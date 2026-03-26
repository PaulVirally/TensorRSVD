TensorRSVD
==========

**TensorRSVD** is a Python library for tensor-free randomized Higher-Order SVD
(HOSVD), also known as the randomized Tucker decomposition.

Rather than storing the tensor as a dense array, TensorRSVD represents it as a
function that returns tensor values at requested coordinates. This makes the
decomposition memory-efficient, i.e., it can handle tensors that are too large
to hold in memory. The library supports three array backends: NumPy (default),
JAX, and CuPy. See :doc:`backends` for details.

**Quick start**

.. code-block:: python

   import numpy as np
   from tensorrsvd import ho_rsvd

   # Tensor defined as a callable: T(x0, x1, x2) = x0 - x1 + x2
   def my_tensor(x0, x1, x2):
       return x0 - x1 + x2

   U_list, S_list = ho_rsvd(
       tensor=my_tensor,
       tensor_shape=(32, 32, 32),
       dtype=np.float64,
       rank=3,
       num_oversamples=10,
       num_power_iterations=2,
       num_idxs=3,
       backend="numpy",
   )

``U_list[m]`` is the :math:`(n_m \times r_m)` factor matrix for mode :math:`m`
and ``S_list[m]`` contains the corresponding mode-:math:`m` singular values. See
the :doc:`User Guide <user_guide>` for a full walkthrough, and the :doc:`Theory
<theory>` page for the underlying mathematics.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   user_guide
   theory
   api_reference
   core
   backends
