import itertools
import numpy as np
import os, sys

import multiprocessing

size = 3000
X = np.random.random((size, size))
result = np.ctypeslib.as_ctypes(np.zeros((size, size)))
shared_array = multiprocessing.RawArray(result._type_, result)



block_size = 250

def fill_subsquare(ll):
    ll_i, ll_j = ll
    tmp = np.ctypeslib.as_array(shared_array)
    coord_pairs_to_fill = itertools.product(range(ll_i, ll_i+block_size), range(ll_j, ll_j+block_size))
    for i, j in coord_pairs_to_fill:
        tmp[i, j] = X[i, j]

window_idxs = itertools.product(range(0, size, block_size), range(0, size, block_size))
import datetime
t0 = datetime.datetime.now()

# for coords in window_idxs:
#     fill_subsquare(coords)

p = multiprocessing.Pool()
for thing in p.imap_unordered(fill_subsquare, window_idxs, chunksize=25):
    pass

result = np.ctypeslib.as_array(shared_array)
t1 = datetime.datetime.now()
print(f"TIME:  {(t1-t0).total_seconds()*1000} ms")
print(np.array_equal(X, result))
