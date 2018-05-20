# -*- coding: utf-8 -*-

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns


class BaseWindow():
    """Base window class."""

    def __init__(self):
        pass

    def plot(self):
        _, ax = plt.subplots(1)
        sns.heatmap(self.matrix.T, vmin=0, vmax=1,
            xticklabels=self.matrix.shape[0]//10,
            yticklabels=self.matrix.shape[1]//10,
            ax=ax)
        ax.invert_yaxis()
        ax.set_title(self.label)
        ax.set_xlabel("query index")
        ax.set_ylabel("reference index")
        plt.show()


class NoWindow(BaseWindow):
    label = "no window"
    def __init__(self, len_x, len_y):
        self._gen_window(len_x, len_y)

    def _gen_window(self, len_x, len_y):
        self.matrix = np.ones([len_x, len_y], dtype=bool)
        self.list = np.argwhere(self.matrix == True)


class SakoechibaWindow(BaseWindow):
    label = "sakoechiba window"
    def __init__(self, len_x, len_y, size):
        self._gen_window(len_x, len_y, size)

    def _gen_window(self, len_x, len_y, size):
        xx = np.arange(len_x)
        yy = np.arange(len_y)
        self.matrix = np.abs(xx[:,np.newaxis] - yy[np.newaxis, :]) <= size
        self.list = np.argwhere(self.matrix == True)


class ItakuraWindow(BaseWindow):
    label = "itakura window"
    def __init__(self, len_x, len_y):
        self._gen_window(len_x, len_y)

    def _gen_window(self, len_x, len_y):
        self.matrix = _gen_itakura_window(len_x, len_y).astype(np.bool)
        self.list = np.argwhere(self.matrix == True)


@jit(nopython=True)
def _gen_itakura_window(len_x, len_y):
    matrix = np.zeros((len_x, len_y), dtype=np.int8)
    for xidx in range(len_x):
        for yidx in range(len_y):
            if (yidx < 2*xidx + 1) and (xidx <= 2*yidx + 1) and \
                (xidx >= len_x - 2*(len_y - yidx)) and \
                (yidx > len_y - 2*(len_x - xidx)):
                matrix[xidx, yidx] = 1
    return matrix


class UserWindow(BaseWindow):
    label = "user defined window"
    def __init__(self, len_x, len_y, win_func, *args, **kwargs):
        """user defined window

        Parameters
        ----------
        len_x : length of query
        len_y : length of reference
        win_func : function
            any function which returns bool
        *args,**kwargs : arguments for win_func

        """
        self._gen_window(len_x, len_y, win_func, *args, **kwargs)

    def _gen_window(self, len_x, len_y, win_func, *args, **kwargs):
        matrix = np.zeros((len_x, len_y), dtype=np.bool)
        for xidx in range(len_x):
            for yidx in range(len_y):
                if win_func(xidx, yidx, *args, **kwargs):
                    matrix[xidx,yidx] = True
        self.matrix = matrix
        self.list = np.argwhere(self.matrix == True)
