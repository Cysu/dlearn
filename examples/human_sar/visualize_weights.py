import os
import sys
import argparse
import numpy as np
import theano

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

from dlearn.visualization import show_channels
from dlearn.utils.serialize import load_data

from scipy.io import loadmat

W = loadmat('first_layer_filter.mat')['W']
M = load_data('mean.pkl')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from skimage.color import lab2rgb

fig = plt.figure()
grid = AxesGrid(fig, 111,
                nrows_ncols=(4, 8),
                axes_pad=0.0,
                share_all=True)

for r in xrange(4):
    for c in xrange(8):
        ind = r * 8 + c
        I = W[ind]
        for i in xrange(3):
            I[i] += M[i].mean()
        I *= 100.0
        I[0] += 35.0
        I[0] = (I[0] - I[0].min()) / (I[0].max() - I[0].min()) * 100.0
        I = np.rollaxis(I, 0, 3)
        I = I.astype('float64')
        I = lab2rgb(I)
        grid[ind].imshow(I)

grid.axes_llc.get_xaxis().set_ticks([])
grid.axes_llc.get_yaxis().set_ticks([])

plt.show()
