import numpy as np


def count(predict, gtruth):
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
