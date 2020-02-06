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
    result_q = multiprocessing.SimpleQueue() # has size limit
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


def globalVar_func(i, total_i, arr, dest_arr, d, v):
    # d and v are dict and int variables, to test their scope
    print(f"Process number {i}, pid {os.getpid()}, starting...")
    # arr is multiprocessing.RawArray
    arr = np.frombuffer(arr)
    sleep(1) # represent work taking some time
    dest_arr = np.frombuffer(dest_arr.get_obj())
    dest_arr[array_slice(i, total_i, arr.size)] = arr[array_slice(i, total_i, arr.size)]*i
    d[i] = arr[i]
    print(f"Process {i} var v={v}")
    v = arr[i]
    print(f"Process {i} var v={v} reassigned")
    print(f"Process number {i}, pid {os.getpid()}, done.")

def globalVarTest():
    """
    Run under if name == main
    This test is meant to confirm how global variables (don't) change when
    used in subprocesses
    """
    procs = []
    n_cpus = 4
    test_dict = dict()
    test_var = None
    test_arr = np.arange(200)
    test_arr_shared = multiprocessing.RawArray('d', test_arr.size)
    test_arr_shared_np = np.frombuffer(test_arr_shared)
    test_arr_shared_np[:] = test_arr
    dest_arr = multiprocessing.Array('d', test_arr.size)
    del test_arr
    print("TEST ARR ORIGINAL")
    print('-'*40)
    print(test_arr_shared_np)
    print('-'*40)
    for i in range(n_cpus):
        p = multiprocessing.Process(target=globalVar_func,
            args=(i, n_cpus, test_arr_shared, dest_arr, test_dict, test_var))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("TEST ARR POST")
    print('-'*40)
    print(test_arr_shared_np)
    print('-'*40)
    print("DESTINATION ARR POST")
    print('-'*40)
    print(np.frombuffer(dest_arr.get_obj()))
    print('-'*40)
    print(f"Variable {test_var}")
    print("Dict:", test_dict)


def arrayDict_func(i, total_i, src_arr_dict, dest_arr_dict):
    # d and v are dict and int variables, to test their scope
    print(f"Process number {i}, pid {os.getpid()}, starting...")
    for k in src_arr_dict:
        arr = np.frombuffer(src_arr_dict[k])
        dest_arr = np.frombuffer(dest_arr_dict[k].get_obj())
        dest_arr[array_slice(i, total_i, arr.size)] = arr[array_slice(i, total_i, arr.size)]*(i+5)
    sleep(1) # represent work taking some time
    print(f"Process number {i}, pid {os.getpid()}, done.")

def arrayDictTest():
    """
    Run under main
    Tests if we can use a "global" dictionary to store shared arrays
    """
    procs = []
    n_cpus = 4
    array_size = 30
    src_arrs, dst_arrs = {}, {}
    for n in range(15, 20):
        k = str(n)
        shared_arr = multiprocessing.RawArray('d', array_size)
        shared_arr_np = np.frombuffer(shared_arr)
        shared_arr_np[:] = np.arange(array_size) + n
        dest_shared_arr = multiprocessing.Array('d', array_size)
        src_arrs[k] = shared_arr
        dst_arrs[k] = dest_shared_arr
    for i in range(n_cpus):
        p = multiprocessing.Process(target=arrayDict_func,
            args=(i, n_cpus, src_arrs, dst_arrs))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    for k in src_arrs:
        print(k, "-"*15, ">>>")
        print(np.frombuffer(src_arrs[k]))
        print('vvv')
        print(np.frombuffer(dst_arrs[k].get_obj()))
        print('-'*15)
        print()
    print()


if __name__ == "__main__":
    arrayDictTest()

"""
Actually I think this is finally making sense
Pool is good for massively parallel but tiny tasks
Process is better for a handful of subprocs that have more complex tasks

Process will let you basically run a completely new, separate thing, just like
running another python command from command line

I think Process is the right approach for parallelizing this code.
"""
