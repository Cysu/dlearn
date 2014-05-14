import numpy as np


def _cmc_core(D, G, P):
    m, n = D.shape
    order = np.argsort(D, axis=0)
    match = (G[order] == P)
    return (match.sum(axis=1) * 1.0 / n).cumsum()


def count(distmat, glabels=None, plabels=None, ds=None, repeat=None):
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
