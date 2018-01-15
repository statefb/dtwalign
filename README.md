# fast-dtw for Python test repository

TODO:
* itakura:unittest
* step_pattern array comment
* plot_path: plot time series if ndim == 1
* fastdtw

## memo
### dependencyに関すること
* seaborn > 0.8.1
* numba 0.22.1はNG?  
0.36.2では動いた on Windows
0.34.0で確認 on mac OSX
###その他
* Rパッケージでは、距離行列をどこで計算しているか？  
→pair-wiseで計算してから
