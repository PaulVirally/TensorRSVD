"""TensorRSVD: tensor-free randomized Higher-Order SVD.

Represents tensors as Python callables and computes their Tucker decomposition
using randomized linear algebra, without ever forming the dense tensor.
The single public entry point is :func:`tensorrsvd.ho_svd_r`.
"""

from importlib.metadata import version

from .api import ho_svd_r

__version__ = version("tensorrsvd")
__all__ = ["ho_svd_r", "__version__"]
