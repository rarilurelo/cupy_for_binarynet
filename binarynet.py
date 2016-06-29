from __future__ import division
from chainer import cuda
import cupy
import time
import numpy as np
import matplotlib.pyplot as plt

def _binarize():
    return cuda.elementwise(
            "T x", "T y",
            "y = x >= 0 ? 1 : 0",
            "binarize")

def _preprocess():
    return cuda.elementwise(
            "raw T x, int32 outdim", "raw T y",
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
            """,
            "preprocess_vec")

def _popcount():
    return cuda.reduce(
            "T x", "T y",
            "2*__popc(x)-32",
            "a+b",
            "y = a",
            "0",
            "popcount")

def _popcount_elem():
    return cuda.elementwise(
            "T x", "T y",
            """
                y = __popc(x)
            """,
            "popcount_elem")
cupy.random.seed(seed=1237)

in_dims = [i*32 for i in range(1, 21)]
out_dims = [i*32 for i in range(1, 21)]

binarize_log       = []
preprocess_log     = []
preprocess_vec_log = []
xnor_log           = []
popcount_log       = []


for in_dim in in_dims:
    for out_dim in out_dims:
        binarize_time       = 0
        preprocess_time     = 0
        preprocess_vec_time = 0
        xnor_time           = 0
        popcount_time       = 0
        
        for _ in range(1):
            W = cupy.random.rand(out_dim, in_dim)-0.5
            x = cupy.random.rand(in_dim, )-0.5
            
            yw = cupy.zeros_like(W)
            yx = cupy.zeros_like(x)
            
            s              = time.time()
            Wb             = _binarize()(W, yw)
            xb             = _binarize()(x, yx)
            Wb             = Wb.astype('int32')
            xb             = xb.astype('int32')
            binarize_time += time.time()-s
            
            s = time.time()
            Wb = _preprocess()(Wb,
                               Wb.shape[0],
                               cupy.zeros((Wb.shape[0], Wb.shape[1]//32)).astype("int32"),
                               size=Wb.shape[1]
                              )
            preprocess_time += time.time()-s
            s = time.time()
            xb = _preprocess_vec()(xb,
                                   cupy.zeros((xb.shape[0]//32)).astype("int32"),
                                   size=xb.shape[0]
                                  )
            preprocess_vec_time += time.time()-s
            
            s          = time.time()
            yb         = cupy.invert(cupy.bitwise_xor(Wb, xb))
            xnor_time += time.time()-s
            
            s              = time.time()
            yb             = _popcount()(yb, axis=1)
            popcount_time += time.time()-s
        
        print "binarize_time: {0}".format(binarize_time)
        print "preprocess_time: {0}".format(preprocess_time)
        print "preprocess_vec_time: {0}".format(preprocess_vec_time)
        print "xnor_time: {0}".format(xnor_time)
        print "popcount_time: {0}".format(popcount_time)

        binarize_log.append(binarize_time)
        preprocess_log.append(preprocess_time)
        preprocess_vec_log.append(preprocess_vec_time)
        xnor_log.append(xnor_time)
        popcount_log.append(popcount_time)



binarize_log       = np.array(binarize_log).reshape(len(in_dims), len(out_dims))
preprocess_log     = np.array(preprocess_log).reshape(len(in_dims), len(out_dims))
preprocess_vec_log = np.array(preprocess_vec_log).reshape(len(in_dims), len(out_dims))
xnor_log           = np.array(xnor_log).reshape(len(in_dims), len(out_dims))
popcount_log       = np.array(popcount_log).reshape(len(in_dims), len(out_dims))

logs = [binarize_log, preprocess_log, preprocess_vec_log, xnor_log, popcount_log]
logs_name = ['binarize_log', 'preprocess_log', 'preprocess_vec_log', 'xnor_log', 'popcount_log']

for i, in_dim in enumerate(in_dims):
    for log, name in zip(logs, logs_name):
        plt.plot(out_dims, log[i], label=name)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.subplots_adjust(right=0.6)
    plt.title("fix in_dim to {0}".format(in_dim))
    plt.xlabel("out_dim size")
    plt.ylabel("time")
    plt.savefig("fix_in_dim_to_{0}.png".format(in_dim))
    plt.clf()

for i, out_dim in enumerate(out_dims):
    for log, name in zip(logs, logs_name):
        plt.plot(in_dims, log[:, i], label=name)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.subplots_adjust(right=0.6)
    plt.legend()
    plt.title("fix out_dim to {0}".format(out_dim))
    plt.xlabel("in_dim size")
    plt.ylabel("time")
    plt.savefig("fix_out_dim_to_{0}.png".format(out_dim))
    plt.clf()






