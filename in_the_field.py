#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 08:36:18 2019

@author: jeremiasjunior
"""

import numpy as np
"""
lista = np.array([[1,2,3],[4,5,6]])

lista_1 = np.array([7,8,9])
lista_3 = np.array([10,11,12])
lista_2 = np.append(lista, [lista_1],axis=0)


lista_3 = np.append(lista_2, [lista_3], axis=0)
print (lista_3)
"""

lista_1 = np.array([])

lista_1 = np.append([1,2,3],lista_1, axis=0)

lista_2 =[3,4,5]

lista_1 = np.append([lista_1], [lista_2], axis = 0)
#lista_3 = np.append(lista_1, lista_2, axis = 0)

print(lista_1)

