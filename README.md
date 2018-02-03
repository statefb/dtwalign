# DTW (Dynamic Time Warping)
Comprehensive dynamic time warping module for python.

## Installation
`pip install dtwalign`

## Features
### Fast computation
---
by [Numba](https://numba.pydata.org)
### Partial alignment
---
  - before alignment

![](img/partial_org.png)

  - after alignment

![](img/partial_res.png)

### Local constraint (step pattern)
---
example:

| Symmetric2 | AsymmetricP2 | TypeIVc |
|:-----------:|:------------:|:------------:|
| ![](img/symmetric2.png) | ![](img/asymmetricP2.png) | ![](img/typeIVc.png) |

### Global constraint (windowing)
---
example:

| Sakoechiba | Itakura | User defined |
|:-----------:|:------------:|:------------:|
| ![](img/sakoechiba.png) | ![](img/itakura.png) | ![](img/user_win.png) |

### Alignment path visualization
---
![](img/partial_path.png)

## Usage
see [example](./example.ipynb)

## Reference
1. Sakoe, H.; Chiba, S., Dynamic programming algorithm optimization for spoken word recognition, Acoustics, Speech, and Signal Processing

* Paolo Tormene, Toni Giorgino, Silvana Quaglini, Mario Stefanelli (2008). Matching Incomplete Time Series with Dynamic Time Warping: An Algorithm and an Application to Post-Stroke Rehabilitation. Artificial Intelligence in Medicine, 45(1), 11-34.

* Toni Giorgino (2009). Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package. Journal of Statistical Software, 31(7), 1-24.
