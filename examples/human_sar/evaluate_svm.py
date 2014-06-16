import os
import sys
from scipy.io import loadmat

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

import dlearn.stats as stats
from dlearn.utils.serialize import save_data


def compute_stats(output, target):
    n = target.shape[1]
    ret = [0] * n
    for j in xrange(n):
        o = output[:, j].ravel()
        t = target[:, j].ravel()
        fpr, tpr, thresh = stats.roc(o, t)
        auc = stats.auc(fpr, tpr)
        ret[j] = (auc, fpr, tpr, thresh)

    return ret


def show_stats(ret):
    import matplotlib.pyplot as plt

    n_cols = 4
    n_rows = len(ret) // n_cols + 1

    for j, (auc, fpr, tpr, thresh) in enumerate(ret):
        # Plot stats
        plt.subplot(n_rows, n_cols, j + 1)
        plt.plot(fpr, tpr)
        plt.title('AUC = {:.2f}%'.format(auc * 100))

    plt.show()


matdata = loadmat('svm_result_mix.mat')
target = matdata['targets']
output = matdata['outputs']

ret = compute_stats(output, target)
save_data(ret, 'stats_attr_svm_mix.pkl')
show_stats(ret)
