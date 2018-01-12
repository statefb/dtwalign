# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BaseWindow():
    def __init__(self):
        pass

    def plot(self):
        _,ax = plt.subplots(1)
        sns.heatmap(self.matrix.T,vmin=0,vmax=1,\
            xticklabels=self.matrix.shape[0]//10,\
            yticklabels=self.matrix.shape[1]//10,\
            ax=ax
        )
        ax.invert_yaxis()
        ax.set_title(self.label)
        ax.set_xlabel("query index")
        ax.set_ylabel("reference index")
        plt.show()


class NoWindow(BaseWindow):
    label = "no window"
    def __init__(self,len_x,len_y):
        self._get_window(len_x,len_y)

    def _get_window(self,len_x,len_y):
        self.matrix = np.ones([len_x,len_y],dtype=bool)
        self.list = np.argwhere(self.matrix == True)

class SakoechibaWindow(BaseWindow):
    label = "sakoechiba window"
    def __init__(self,len_x,len_y,size):
        self._get_window(len_x,len_y,size)

    def _get_window(self,len_x,len_y,size):
        xx = np.arange(len_x)
        yy = np.arange(len_y)
        self.matrix = np.abs(xx[:,np.newaxis] - yy[np.newaxis,:]) < size
        self.list = np.argwhere(self.matrix == True)

class ItakuraWindow(BaseWindow):
    # TODO
    label = "Itakura window"
    def __init__(self):
        raise NotImplementedError()

class UserWindow(BaseWindow):
    # TODO
    label = "user difined window"
    def __init__(self):
        raise NotImplementedError()
