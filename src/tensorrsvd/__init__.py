"""TensorRSVD: tensor-free randomized Higher-Order SVD.

Represents tensors as Python callables and computes their Tucker decomposition
using randomized linear algebra, without ever forming the dense tensor.
The single public entry point is :func:`tensorrsvd.ho_rsvd`.
"""

from importlib.metadata import version

from .api import ho_rsvd

__version__ = version("tensorrsvd")
__all__ = ["ho_rsvd", "__version__"]
