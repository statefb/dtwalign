# -*- coding: utf-8 -*-
"""Cost matrix computation."""

import numpy as np
from numba import jit


@jit(nopython=True)
def _calc_cumsum_matrix_jit(X, w_list, p_ar, open_begin):
    """Fast implementation by numba.jit."""
    len_x, len_y = X.shape
    # cumsum matrix
    D = np.ones((len_x, len_y), dtype=np.float64) * np.inf

    if open_begin:
        X = np.vstack((np.zeros((1, X.shape[1])), X))
        D = np.vstack((np.zeros((1, D.shape[1])), D))
        w_list[:, 0] += 1

    # number of patterns
    num_pattern = p_ar.shape[0]
    # max pattern length
    max_pattern_len = p_ar.shape[1]
    # pattern cost
    pattern_cost = np.zeros(num_pattern, dtype=np.float64)
    # step cost
    step_cost = np.zeros(max_pattern_len, dtype=np.float64)
    # number of cells
    num_cells = w_list.shape[0]

    for cell_idx in range(num_cells):
        i = w_list[cell_idx, 0]
        j = w_list[cell_idx, 1]
        if i == j == 0:
            D[i, j] = X[0, 0]
            continue

        for pidx in range(num_pattern):
            # calculate local cost for each pattern
            for sidx in range(1, max_pattern_len):
                # calculate step cost of pair-wise cost matrix
                pattern_index = p_ar[pidx, sidx, 0:2]
                ii = int(i + pattern_index[0])
                jj = int(j + pattern_index[1])
                if ii < 0 or jj < 0:
                    step_cost[sidx] = np.inf
                    continue
                else:
                    step_cost[sidx] = X[ii, jj] \
                        * p_ar[pidx, sidx, 2]

            pattern_index = p_ar[pidx, 0, 0:2]
            ii = int(i + pattern_index[0])
            jj = int(j + pattern_index[1])
            if ii < 0 or jj < 0:
                pattern_cost[pidx] = np.inf
                continue

            pattern_cost[pidx] = D[ii, jj] \
                + step_cost.sum()

        min_cost = pattern_cost.min()
        if min_cost != np.inf:
            D[i, j] = min_cost

    return D
