import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


def show_channels(chmaps, n_cols=8):
    n_rows = (chmaps.shape[0] - 1) // n_cols + 1

    grid = AxesGrid(plt.figure(), 111,
                    nrows_ncols=(n_rows, n_cols),
                    axes_pad=0.0,
                    share_all=True)

    for i, chmap in enumerate(chmaps):
        grid[i].imshow(chmap, vmin=chmaps.min(), vmax=chmaps.max())

    grid.axes_llc.get_xaxis().set_ticks([])
    grid.axes_llc.get_yaxis().set_ticks([])

    plt.draw()
    plt.show()
