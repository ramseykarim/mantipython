import os, sys
import multiprocessing
from time import sleep
import numpy as np

"""
A (less confusing) successor to the v2/multiprocessingTest
Remember to comment all your test code so you can return to it
and use it again :-)
Created January 29, 2019
"""
__author__ = "Ramsey Karim"

def simple_parallel_func(t):
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid(), "starting")
    sleep(t)
    print('process id:', os.getpid(), "done sleeping")

def pfuncTest():
    """
    Run this under 'if name main'
    """
    procs = []
    for i in range(4):
        p = multiprocessing.Process(target=simple_parallel_func, args=(4,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    simple_parallel_func(1)

def array_parallel_func(i, total_i, arr):
    raise NotImplementedError("not finished writing!!!")
    print(f"Process number {i}, pid {os.getpid()}, starting...")
    arr_slice_size = int(arr.size//total_i) + 1
    arr = arr[slice(arr_slice_size*i, arr_slice_size*(i+1))]
    ##### CONTINUE LATER

if __name__ == "__main__":
    array_parallel_func(None, None, None)

"""
Actually I think this is finally making sense
Pool is good for massively parallel but tiny tasks
Process is better for a handful of subprocs that have more complex tasks

Process will let you basically run a completely new, separate thing, just like
running another python command from command line

I think Process is the right approach for parallelizing this code.
"""
