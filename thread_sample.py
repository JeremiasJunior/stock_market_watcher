#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 00:52:16 2019

@author: jeremiasj
"""

import numpy as np
import threading
import time
start_time = time.time()


def do_something():
    time.sleep(1)
    print('done sleeping')

t1 = threading.Thread(target=do_something)
t2 = threading.Thread(target=do_something)