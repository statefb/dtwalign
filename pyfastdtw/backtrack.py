# -*- coding: utf-8 -*-

import numpy as np
from numba import jit

@jit(nopython=True)
def _backtrack_jit(D):
    pass

def _backtrack_py(D,pattern):
    """
    naive implementation
    """
    i,j = D.shape
    i -= 1
    j -= 1
    # path
    path = [(i,j)]
    # pattern array
    p_ar = pattern.array

    D_cache = np.ones(pattern.num_pattern) * np.inf
    while not (i == 0 and j == 0):
        path_cache = []
        for pidx in range(pattern.num_pattern):
            # get D value corresponds to end of pattern
            pattern_index = p_ar[pidx,0,0:2]
            ii = int(i + pattern_index[0])
            jj = int(j + pattern_index[1])
            if ii < 0 and jj < 0:
                D_cache[pidx] = np.inf
            else:
                D_cache[pidx] = D[ii,jj]

            path_step = []
            for sidx in range(pattern.max_pattern_len)[::-1]:
                # memorize where arrived
                if p_ar[pidx,sidx,2] == 0:
                    # if weight value is 0, the row is padded-row
                    continue
                pattern_index = p_ar[pidx,sidx,0:2]
                if pattern_index[0] == 0 and pattern_index[1] == 0:
                    continue
                ii = int(i + pattern_index[0])
                jj = int(j + pattern_index[1])
                path_step.append((ii,jj))
            path_cache.append(path_step)

        # find path minimize D_chache
        # print("D_cache:{},path_cache:{}".format(D_cache,path_cache))
        min_pattern_idx = np.argmin(D_cache)
        path += path_cache[min_pattern_idx]
        i += p_ar[min_pattern_idx,0,0]
        j += p_ar[min_pattern_idx,0,1]
        if i < 0: i = 0
        if j < 0: j = 0

    path.reverse()
    path = np.array(path)
    return path
