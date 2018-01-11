# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns

class BaseWindow():
    def __init__(self):
        pass

    def plot(self):
        plt.figure()
        sns.heatmap(self.matrix,vmin=0,vmax=1)
        plt.title(self.label)
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
