import threading

"""
Learning the threading package

The test case will be:
Draw data from 5 large, same-shape arrays and process it, filling in a
results array
Needs to access several global variables
"""

import numpy as np
import matplotlib.pyplot as plt

dsize = 1000
target = np.full(dsize, np.nan)

target_dicts = []
lock = threading.Lock()

def func(thread_index, number_of_threads):
    source = [np.arange(dsize)[:, np.newaxis] for i in range(5)]
    support_1 = np.arange(100000)[np.newaxis, :]
    support_2 = [i+1 for i in range(5)]
    support_3 = 2.5
    block_size = dsize//number_of_threads
    end_idx = min(block_size*(thread_index+1), dsize)
    if thread_index == number_of_threads - 1:
        end_idx = dsize
    target_indices = np.array(range(block_size*thread_index, end_idx))
    # target_indices = range(thread_index, dsize, number_of_threads)
    thread_dict = np.full(len(target_indices), np.nan)
    lock.acquire()
    target_dicts.append((target_indices, thread_dict))
    lock.release()
    for meta_i, i in enumerate(target_indices):
        total = np.sum([np.sum(x[i]*support_1/np.sqrt(support_1+x[i]+1))*y for x, y in zip(source, support_2)])
        total *= support_3
        # target[i] = thread_index
        thread_dict[meta_i] = thread_index
    lock.acquire()
    target[(target_indices,)] = thread_dict
    lock.release()
    print(f"{thread_index} done!")

import datetime

def no_parallel():
    t0 = datetime.datetime.now()
    func(0, 1)
    t1 = datetime.datetime.now()
    print(f"TIME ({dsize}):  {(t1-t0).total_seconds()*1000} ms")
    print("worked" if not np.any(np.isnan(target)) else "didn't work")
    plt.plot(target, '.')
    plt.show()


def simple_parallel():
    n_threads = 6
    threads = []
    t0 = datetime.datetime.now()
    for t_idx in range(n_threads):
        t = threading.Thread(target=func, args=(t_idx, n_threads))
        threads.append(t)
        print(f"{t.name} ({t_idx}) starting!")
        t.start()
    for t_idx, t in enumerate(threads):
        t.join()
        print(f"{t.name} ({t_idx}) joined!")

    t1 = datetime.datetime.now()
    print(f"TIME ({dsize}):  {(t1-t0).total_seconds()*1000} ms")
    print("worked" if not np.any(np.isnan(target)) else "didn't work")
    plt.plot(target, '.')
    plt.show()


# no_parallel()
simple_parallel()
