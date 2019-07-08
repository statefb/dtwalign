========
Tutorial
========

Basic Usage
===========

Firstly, let's generate toy data for this tutorial.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(1234)
    # generate toy data
    x = np.sin(2*np.pi*3.1*np.linspace(0,1,101))
    x += np.random.rand(x.size)
    y = np.sin(2*np.pi*3*np.linspace(0,1,120))
    y += np.random.rand(y.size)

    plt.plot(x,label="query")
    plt.plot(y,label="reference")
    plt.legend()
    plt.ylim(-1,3)

.. image:: img/tutorial/toy_data.png

Then run dtw method which returns DtwResult object.

.. code-block:: python

    from dtwalign import dtw
    res = dtw(x,y)

.. note::
    The first run takes a few seconds for jit compilation.

dtw distance
------------

dtw distance can be refered via DtwResult object.

.. code-block:: python

    print("dtw distance: {}".format(res.distance))
    # dtw distance: 30.048812654583166
    print("dtw normalized distance: {}".format(res.normalized_distance))
    # dtw normalized distance: 0.13596747807503695

.. note::
    If you want to calculate only dtw distance (i.e. no need to gain alignment path),
    give 'distance_only' argument as True when runs `dtw` method (it makes faster).

alignment path
--------------

DtwResult object offers a method which visualize alignment path with cumsum cost matrix.

.. code-block:: python

    res.plot_path()

.. image:: img/tutorial/alignment_path.png

Alignment path array also provided:

.. code-block:: python

    x_path = res.path[:,0]
    y_path = res.path[:,1]

warp one to the other
---------------------

`get_warping_path` method provides an alignment path of X with fixed Y and vice versa.

.. code-block:: python

    # warp x to y
    x_warping_path = res.get_warping_path(target="query")
    plt.plot(x[x_warping_path],label="aligned query to reference")
    plt.plot(y,label="reference")
    plt.legend()
    plt.ylim(-1,3)

.. image:: img/tutorial/x_to_y.png

.. code-block:: python

    # warp y to x
    y_warping_path = res.get_warping_path(target="reference")
    plt.plot(x,label="query")
    plt.plot(y[y_warping_path],label="aligned reference to query")
    plt.legend()
    plt.ylim(-1,3)

.. image:: img/tutorial/y_to_x.png

Advanced Usage
==============
global constraint
-----------------
regarding how to run dtw with global constrained which is also called windowing.

local constraint
----------------

regarding how to run dtw with local constrained which is also called step pattern.

partial alignment
-----------------

regarding how to perform partial matching algorithm.

use other metric
----------------

how to use other pair-wise distance metric (default is euclidean).

use pre-computed distance matrix
--------------------------------

how to run dtw with given pre-computed distance matrix, not with X and Y.

use user-defined constraints
----------------------------

how to define user constraint and to use.

Utilities
=========