import numpy as np
import cupy
from chainer import cuda

def _mul_i():
    return cuda.elementwise(
            "raw T x", "T y",
            "y = x[i]*i",
            "muli")


o = cupy.ones((3,2))
print _mul_i()(o)
