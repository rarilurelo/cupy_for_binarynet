import numpy as np
import cupy
from chainer import cuda

def _mul_i():
    return cuda.elementwise(
            "raw T x", "raw T y",
            """
                y[i] = x[i]
            """,
            "muli")

o = cupy.ones((3,2,2))
y = cupy.zeros_like(o)
print _mul_i()(o,y, size=6)
