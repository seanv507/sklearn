# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 18:15:43 2014
http://mail.scipy.org/pipermail/scipy-user/2013-June/034744.html

"""
import numpy as np
from math import log, floor, ceil, fmod

def halton(dim, nbpts):
    h = np.empty(nbpts * dim)
    h.fill(np.nan)
    p = np.empty(nbpts)
    p.fill(np.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(ceil(lognbpts / log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1) )

        for j in range(nbpts):
            d = j + 1
            sum_ = fmod(d, b) * p[0]
            for t in range(1, n):
                d = floor(d / b)
                sum_ += fmod(d, b) * p[t]
            h[j*dim + i] = sum_

    return h.reshape(nbpts, dim)
