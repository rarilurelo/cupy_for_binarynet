from __future__ import division
from chainer import cuda
import cupy

def _binarize():
    return cuda.elementwise(
            "T x", "T y",
            "y = x >= 0 ? 1 : 0",
            "binarize")

def _preprocess():
    return cuda.elementwise(
            "raw T x, int32 indim", "raw T y",
            """
                int max = _ind.size()/32;
                for(int j = 0; j < max; j++){
                    int ind_y[] = {i, j};
                    y[ind_y] = 0;
                    for(int k = 0; k < 32; k++){
                        int ind[] = {i, j+k};
                        y[ind_y] = y[ind_y] | (x[ind] << k);
                    }
                }
                int ind_y[] = {i, max};
                y[ind_y];
                for(int j = max*32; j < indim; j++){
                    int ind[] = {i, j};
                    y[ind_y] = y[ind_y] | (x[ind] << j);
                }
            """,
            "preprocess")

def _preprocess_vec():
    return cuda.elementwise(
            "raw T x", "raw T y",
            """
                int max = _ind.size()/32;
                for(int j = 0; j < max; j++){
                    for(int k = 0; k < 32; k++){
                        y[j] = y[j] | (x[j+k] << k);
                    }
                }
                for(int j = max*32; j < _ind.size(); j++){
                    y[max] = y[max] | (x[j] << j);
                }
            """,
            "preprocess_vec")


def _popcount():
    return cuda.reduce(
            "T x", "T y",
            "__popc(x)",
            "a+b",
            "y = a",
            "0",
            "popcount")


W = cupy.array([[-1,-2,-3,-3,4],[-1,2,2,1,2],[0,1,-1,-1,-1]]).astype("int32")
x = cupy.array([-1,2,3,-1,1]).astype("int32")

yw = cupy.zeros_like(W)
yx = cupy.zeros_like(x)

Wb = _binarize()(W, yw)
xb = _binarize()(x, yx)

print "binarize"
print Wb
print xb

Wb = _preprocess()(Wb, Wb.shape[1], cupy.zeros((Wb.shape[0], Wb.shape[1]//32+1)).astype("int32"), size=Wb.shape[0])
xb = _preprocess_vec()(xb, cupy.zeros((xb.shape[0]//32+1)).astype("int32"), size=xb.shape[0])

print "preprocess"
print Wb
print xb

yb = cupy.invert(cupy.bitwise_xor(Wb, xb))

print "xnor"
print yb

yb = _popcount()(yb, axis=1)

print "popcount"
print yb





