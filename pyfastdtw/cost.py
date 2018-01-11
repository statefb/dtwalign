# -*- coding: utf-8 -*-

import numpy as np
from numba import jit

@jit
def _calc_cumsum_matrix(X,window,pattern):
    """
    calculate cumsum matrix
    """
    len_x,len_y = X.shape
    # cumsum matrix
    D = np.zeros([len_x,len_y])


def _calc_cumsum_matrix_py(X,window,pattern):
    """
    naive implementation
    input:
        X: pair-wise cost matrix
        window: Window instance
        pattern: Pattern instance
    """
    len_x,len_y = X.shape
    # cumsum matrix
    D = np.ones([len_x,len_y]) * np.inf
    # pattern array
    p_ar = pattern.array
    # pattern cost
    pattern_cost = np.zeros(pattern.num_pattern)
    # sequence cost
    step_cost = np.zeros(pattern.max_pattern_len)

    for i,j in window.list:
        if i == j == 0:
            D[i,j] = X[0,0]
            continue

        for pidx in range(pattern.num_pattern):
            # calculate local cost for each pattern
            for sidx in range(1,pattern.max_pattern_len):
                # calculate step cost of pair-wise cost matrix
                pattern_index = p_ar[pidx,sidx,0:2]
                ii = i + pattern_index[0]
                jj = j + pattern_index[1]
                if ii < 0 or jj < 0:
                    step_cost[sidx] = np.inf
                    continue
                else:
                    step_cost[sidx] = X[ii,jj] \
                        * p_ar[pidx,sidx,2]

            pattern_index = p_ar[pidx,0,0:2]
            ii = i + pattern_index[0]
            jj = j + pattern_index[1]
            if ii < 0 or jj < 0:
                pattern_cost[pidx] = np.inf
                continue

            pattern_cost[pidx] = D[ii,jj] \
                + step_cost.sum()

        D[i,j] = pattern_cost.min()
    return D
