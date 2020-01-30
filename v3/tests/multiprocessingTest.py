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
    # Just test what happens when I 'sleep'
    # Mainly to see when things happen in serial/parallel
    # Do some soundouff. Following code copied from the internet
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid(), "starting")
    sleep(t) # Do the actual sleeping
    print('process id:', os.getpid(), "done sleeping")

def simpleTest():
    """
    Run this under 'if name == main'
    """
    procs = []
    for i in range(4):
        p = multiprocessing.Process(target=simple_parallel_func, args=(2,))
        p.start()
        procs.append(p)
        # If I did p.join() here, this would essentially be serial
    # Now, procs are running in parallel
    for p in procs:
        p.join()
    simple_parallel_func(1) # Does not have a parent PID

def array_slice(i, total_i, array_size):
    arr_slice_size = int(array_size//total_i) + 1
    return slice(arr_slice_size*i, arr_slice_size*(i+1))

def array_parallel_func(i, total_i, arr, q):
    print(f"Process number {i}, pid {os.getpid()}, starting...")
    arr = arr[array_slice(i, total_i, arr.size)]
    sleep(1) # represent work taking some time
    q.put((i, arr*2.))
    print(f"Process number {i}, pid {os.getpid()}, done.")

def arrayTest():
    """
    Run under if name == main
    """
    procs = []
    n_cpus = 4
    result_q = multiprocessing.SimpleQueue()
    test_arr = np.arange(61)
    test_arr_size = test_arr.size
    print('-'*40)
    print(test_arr)
    print('-'*40)
    for i in range(n_cpus):
        p = multiprocessing.Process(target=array_parallel_func,
            args=(i, n_cpus, test_arr, result_q))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    while not result_q.empty():
        i, arr_subset = result_q.get()
        test_arr[array_slice(i, n_cpus, test_arr_size)] = arr_subset
    print('-'*40)
    print(test_arr)
    print('-'*40)


if __name__ == "__main__":
    arrayTest()

"""
Actually I think this is finally making sense
Pool is good for massively parallel but tiny tasks
Process is better for a handful of subprocs that have more complex tasks

Process will let you basically run a completely new, separate thing, just like
running another python command from command line

I think Process is the right approach for parallelizing this code.
"""
