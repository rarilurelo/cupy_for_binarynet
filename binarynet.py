import numpy as np
from chainer import cuda

def _binarize():
    return cuda.elementwise(
            "T x", "T y",
            "y = x >= 0 ? 1 : -1",
            "binarize")

def _preprocess():
    return cuda.elementwise(
            "raw T x", "T y",
            """
                x = (x >= 0);
                y = x[i] << i
            """,
            "preprocess")

def _reduction():
    return cuda.reduce(
            "T x", "T y",
            "x",
            "a | b",
            "y = a",
            "0",
            "reduction")

def _xnor():
    return cuda.elementwise(
            "")







