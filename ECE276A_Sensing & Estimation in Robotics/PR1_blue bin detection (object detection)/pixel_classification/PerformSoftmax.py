# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 22:21:14 2022

@author: Yen
"""

import math
import numpy as np

def perform_softmax(v):
    '''
    Perform softmax on matrix v.

    Parameters
    ----------
    v : Numpy.ndarray
        Each row represnets a complete set for softmax.

    Returns
    -------
    v : Numpy.ndarray
        Result of applying softmax on each row.

    '''
    m = np.shape(v)[0]
    n = np.shape(v)[1]
    for i in range(m):
        acc = 0
        for j in range(n):
            v[i][j] = math.exp(v[i][j])
            acc += v[i][j]
        v[i] = v[i] / acc
    return v