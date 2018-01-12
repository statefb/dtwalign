# fast-dtw for Python test

TODO:
* warp function (output warp index)
* openend,openbigin
* fastdtw

## memo
* seaborn > 0.8.1?  
それ以下だとヒートマップにパスを重ね合わせた時に反転してしまう？
* numba 0.22.1はNG?  
0.36.2では動いた on Windows
* Rパッケージでは、距離行列をどこで計算しているか？  
累積行列計算時に逐次計算？それともpair-wiseで計算してから？
