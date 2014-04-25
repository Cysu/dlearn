import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


def show_channels(chmaps, n_cols=8, normalize=None):
    """Display multiple channels of 2D images.

    Parameters
    ----------
    chmaps : numpy.ndarray like
        The channel maps to be displayed. The shape of `chmaps` should be
        (n_channels, height, width).
    n_cols : int, optional
        The number of channels to be displayed in each row. Default is 8.
    normalize : None or list/tuple of int, optional
        If None, each channel will be normalized itself, otherwise all the
        channels will be uniformly normalized according to `normalize`, which
        should be (vmin, vmax). Default is None.

    """
    n_rows = (chmaps.shape[0] - 1) // n_cols + 1

    if n_rows == 1:
        n_cols = chmaps.shape[0]

    if normalize is None:
        vmin, vmax = None, None
    else:
        vmin, vmax = normalize

    grid = AxesGrid(plt.figure(), 111,
                    nrows_ncols=(n_rows, n_cols),
                    axes_pad=0.0,
                    share_all=True)

    for i, chmap in enumerate(chmaps):
        grid[i].imshow(chmap, vmin=vmin, vmax=vmax)

    grid.axes_llc.get_xaxis().set_ticks([])
    grid.axes_llc.get_yaxis().set_ticks([])

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
