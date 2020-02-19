#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:28:41 2020

@author: gbson
"""
def Multiply(num1, num2):
    answer = num1 * num2
    return answer



lista = [10,15,20,25,30,35]
argumento = [12,53,21,98]





list = [lista, argumento]
'''
loops = []
for i in range(0,len(list)):
    print(i)
    loops.append(len(list[i]))
'''
prod = []
res = 0
n_loops = []
for i in range(0,len(list)):
    if res == 0:
        res = len(list[0])
    if i + 1 != len(list):
        res_prod = Multiply(res,len(list[i + 1]))
        res = res_prod

n_loops = res

for i in range(0,n_loops):
    