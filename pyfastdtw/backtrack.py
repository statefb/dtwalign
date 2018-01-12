# -*- coding: utf-8 -*-

import numpy as np
from numba import jit

@jit(nopython=True)
def _backtrack_jit(D,p_ar):
        """
        fast implementation by numba.jit
        """
        # number of patterns
        num_pattern = p_ar.shape[0]
        # max pattern length
        max_pattern_len = p_ar.shape[1]

        i,j = D.shape
        i -= 1
        j -= 1
        # path
        # path = np.ones((sum(D.shape),2),dtype=np.int64) * -1
        # path[0,:] = (i,j)
        path = np.array(((i,j),),dtype=np.int64)
        # cache to memorize path
        path_cache = np.ones((num_pattern,max_pattern_len,2),dtype=np.int64) * -1
        # cache to memorize D
        D_cache = np.ones(num_pattern,dtype=np.float64) * np.inf

        while not (i == 0 and j == 0):
            for pidx in range(num_pattern):
                # get D value corresponds to end of pattern
                pattern_index = p_ar[pidx,0,0:2]
                ii = int(i + pattern_index[0])
                jj = int(j + pattern_index[1])
                if ii < 0 and jj < 0:
                    D_cache[pidx] = np.inf
                else:
                    D_cache[pidx] = D[ii,jj]

                weight_col = p_ar[pidx,:,2]
                step_selector = weight_col != 0

                # initialize cache by NA
                path_step = np.ones((max_pattern_len,2),\
                    dtype=np.int64) * -1
                for sidx in np.where(step_selector)[0]:
                    # memorize where arrived
                    # import pdb; pdb.set_trace()
                    pattern_index = p_ar[pidx,sidx,0:2]
                    if pattern_index[0] == 0 and pattern_index[1] == 0:
                        # note: starting point of pattern was already added
                        continue
                    ii = int(i + pattern_index[0])
                    jj = int(j + pattern_index[1])

                    path_step[sidx,:] = (ii,jj)
                # memorize
                path_cache[pidx,:,:] = path_step
                # import pdb; pdb.set_trace()

            # find path minimize D_chache
            # print("D_cache:{},path_cache:{}".format(D_cache,path_cache))
            min_pattern_idx = np.argmin(D_cache)
            path_to_add = path_cache[min_pattern_idx,:,:]

            # omit NA
            selector = path_to_add.sum(axis=1) != -2
            path_to_add = path_to_add[np.where(selector)[0],:][::-1]

            path = np.vstack((path,path_to_add))

            i += p_ar[min_pattern_idx,0,0]
            j += p_ar[min_pattern_idx,0,1]
            if i < 0: i = 0
            if j < 0: j = 0

        return path[::-1]

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
                    # note: starting point of pattern was already added
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
