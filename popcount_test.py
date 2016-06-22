import cupy
from chainer import cuda



def _popcount():
    return cuda.reduce(
            "T x", "T y",
            "__popc(x)",
            "a+b",
            "y = a",
            "0",
            "popcount")


a = cupy.array([[3,1,0],[1,0,0]])
print _popcount()(a, axis=0)
