import itertools
import numpy as np
import os, sys

import multiprocessing

"""
Wrote this around Dec 13 2019 and returning to it on January 29 2020
to see if I can still salvage useful bits of multiprocessing

Commenting my way through this script to remind myself what I did
"""
__author__ = "Ramsey Karim"

# One side length, so size**2 pixels total
size = 3000
# Whether or not to use memmap, which will put this thing on disk
memmap = False
if memmap:
    np.random.random((size, size)).tofile('data.dat')
    # mode='r' is open existing file for read only
    X = np.memmap('data.dat', dtype=np.float, mode='r', shape=(size, size))
else:
    X = np.random.random((size, size))
# Set up ctypes / RawArray for multiproc???? Why???
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

# What the hell is going on here
p = multiprocessing.Pool()
for thing in p.imap_unordered(fill_subsquare, window_idxs, chunksize=25):
    pass

result = np.ctypeslib.as_array(shared_array)
t1 = datetime.datetime.now()
print(f"TIME:  {(t1-t0).total_seconds()*1000} ms")
print(np.array_equal(X, result))
if memmap:
    os.remove('data.dat')
