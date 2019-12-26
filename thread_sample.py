#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 00:52:16 2019

@author: jeremiasj
"""

import numpy as np
import threading
import time
import concurrent.futures

start = time.perf_counter()


def do_something(seconds):
    print('Sleeping...')
    time.sleep(seconds)
    return 'Done Sleeping...'


with concurrent.futures.ThreadPoolExecutor() as executor:  #context management
    result = [executor.submit(do_something, 1) for _ in range(10)]
    for f in concurrent.futures.as_completed(result):
        print(f.result())

threads = []

#for _ in range(100):
    #t = threading.Thread(target=do_something, args=[2])
    #t.start()
    #threads.append(t)

#for thread in threads:
#    thread.join()
    
finish = time.perf_counter()

print(f'finished in {round(finish-start,2)} seconds ')