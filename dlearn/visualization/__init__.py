import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import AxesGrid


def show_channels(chmaps, n_cols=8, normalize=None, grayscale=False, ofpath=None):
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
        channels will be uniformly normalized according to this argument, which
        should be (vmin, vmax). Default is None.
    grayscale : bool, optional
        If False, each channel will be displayed as color heat map. Otherwise
        each channel will be displayed as grayscale image. Default is False.
    ofpath : None or str, optional
        If None, then the figure will be plotted to a window. Otherwise the
        figure will be saved to `ofpath`. Default is None.

    """
    n_rows = (chmaps.shape[0] - 1) // n_cols + 1

    if n_rows == 1:
        n_cols = chmaps.shape[0]

    if normalize is None:
        vmin, vmax = None, None
    else:
        vmin, vmax = normalize

    fig = plt.figure()

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(n_rows, n_cols),
                    axes_pad=0.0,
                    share_all=True)

    if not grayscale:
        for i, chmap in enumerate(chmaps):
            grid[i].imshow(chmap, vmin=vmin, vmax=vmax)
    else:
        for i, chmap in enumerate(chmaps):
            grid[i].imshow(chmap, cmap=cm.Greys_r)

    grid.axes_llc.get_xaxis().set_ticks([])
    grid.axes_llc.get_yaxis().set_ticks([])

    if ofpath is None:
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()
    else:
        fig.savefig(ofpath)
        plt.close(fig)
