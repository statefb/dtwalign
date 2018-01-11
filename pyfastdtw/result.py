# -*- coding: utf-8 -*-

class DtwResult():
    def __init__(self,cumsum_matrix,path,window,pattern):
        self._cumsum_matrix = cumsum_matrix
        self._window = window
        self._pattern = pattern
        # alignment distance
        self.distance = cumsum_matrix[-1,-1]

        if self._pattern.is_normalizable:
            # normalized cumsum matrix
            self._normalized_cumsum_matrix = \
                self._pattern.normalize_cumsum_matrix(cumsum_matrix)
            # normalized distance
            self.normalized_distance = self._normalized_cumsum_matrix[-1,-1]

    def plot_window(self):
        self._window.plot()

    def plot_path(self):
        pass

    def plot_pattern(self):
        self._pattern.plot()
