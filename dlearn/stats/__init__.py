import numpy as np


def _cmc_core(D, G, P):
    m, n = D.shape
    order = np.argsort(D, axis=0)
    match = (G[order] == P)
    return (match.sum(axis=1) * 1.0 / n).cumsum()


def cmc(distmat, glabels=None, plabels=None, ds=None, repeat=None):
    """Compute the Cumulative Match Characteristic (CMC)

    This function assumes that gallery labels have no duplication. If there are
    duplications, random downsampling will be performed on gallery labels, and
    the computation will be repeated to get an average result.

    Parameters
    ----------
    distmat : numpy.ndarray
        The distance matrix. ``distmat[i, j]`` is the distance between i-th
        gallery sample and j-th probe sample.
    glabels : numpy.ndarray or None, optional
    plabels : numpy.ndarray or None, optional
        If None, then gallery and probe labels are assumed to have no
        duplications. Otherwise, they represent the vector of gallery and probe
        labels. Default is None.
    ds : int or None, optional
        If None, then no downsampling on gallery labels will be performed.
        Otherwise, it represents the number of gallery labels to be randomly
        selected. Default is None.
    repeat : int or None, optional
        If None, then the function will repeat the computation for 100 times
        when downsampling is performed. Otherwise, it specifies the number of
        repetition. Default is None.

    Returns
    -------
    out : numpy.ndarray
        The rank-1 to rank-m accuracy, where m is the number of (downsampled)
        gallery labels.

    """
    m, n = distmat.shape

    if glabels is None and plabels is None:
        glabels = np.arange(0, m)
        plabels = np.arange(0, n)

    if isinstance(glabels, list):
        glabels = np.asarray(glabels)

    if isinstance(plabels, list):
        plabels = np.asarray(plabels)

    ug = np.unique(glabels)

    if repeat is None:
        repeat = 1 if ds is None else 100

    if ds is None:
        ds = ug.size

    ret = 0
    for __ in xrange(repeat):
        # Randomly select gallery labels
        ind = np.sort(np.random.choice(ug.size, ds, replace=False))
        G = ug[ind]

        # Select corresponding probe samples
        ind = []
        for i, label in enumerate(plabels):
            if label in G:
                ind.append(i)
        P = plabels[ind]

        # Randomly select one gallery sample per label selected
        D = np.zeros((ds, P.size))
        for i, g in enumerate(G):
            samples = np.where(G == g)[0]
            j = np.random.choice(samples)
            D[i, :] = distmat[j, ind]

        # Compute CMC
        ret += _cmc_core(D, G, P)

    return ret / repeat


def roc(predict, gtruth):
    """Compute the Receiver Operating Characteristic (ROC)

    Parameters
    ----------
    predict : numpy.ndarray
        The predict scores vector. The score range must be in :math:`[0, 1]`.
    gtruth : numpy.ndarray
        The ground truth labels vector. The label must be either 0 or 1.

    Returns
    -------
    fpr, tpr, thresh : numpy.ndarray
        The false positive rate, true positive rate, and corresponding score
        threshold. The predict label is :math:`\delta(score > threshold)`.

    """

    n = gtruth.size
    n_pos = gtruth.sum()
    n_neg = n - n_pos

    ind = np.argsort(predict)
    predict = predict[ind]
    gtruth = gtruth[ind]

    fn = np.insert(gtruth.cumsum(dtype=np.float32), 0, 0)

    fp = n_neg - (np.arange(n + 1) - fn)
    tp = n_pos - fn

    fpr = fp / n_neg
    tpr = tp / n_pos

    thresh = np.insert(predict, 0, 0)

    return (fpr[::-1], tpr[::-1], thresh[::-1])


def auc(x, y, range_x=None, range_y=None):
    """Compute the Area Under Curve (AUC)

    Parameters
    ----------
    x : numpy.ndarray
        The vector of x values.
    y : numpy.ndarray
        The vector of y values.
    range_x : tuple of float or None, optional
        If None, then the first and last element of x will be used as the
        range of x. Otherwise, it is a tuple (xmin, xmax). Default is None.
    range_y : tuple of float or None, optional
        If None, then the first and last element of y will be used as the range
        of y. Otherwise, it is a tuple (ymin, ymax). Default is None.

    Returns
    -------
    out : float
        The area under curve over the total area.

    """
    if range_x is None:
        range_x = (x.min(), x.max())

    if range_y is None:
        range_y = (y.min(), y.max())

    auc = 0
    for i in xrange(len(x) - 1):
        dx = x[i + 1] - x[i]
        ybar = (y[i + 1] + y[i]) / 2.0
        auc += ybar * dx

    return auc / ((range_y[1] - range_y[0]) * (range_x[1] - range_x[0]))
