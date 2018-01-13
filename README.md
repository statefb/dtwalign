# fast-dtw for Python test

TODO:
* warp function (output warp index)
* openend,openbigin
* fastdtw

## memo
### dependencyに関すること
* seaborn > 0.8.1?  
それ以下だとヒートマップにパスを重ね合わせた時に反転してしまう？
* numba 0.22.1はNG?  
0.36.2では動いた on Windows
0.34.0で確認 on mac OSX
###その他
* Rパッケージでは、距離行列をどこで計算しているか？  
→pair-wiseで計算してから
