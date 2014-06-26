import os
import sys
import argparse
import numpy as np
import theano

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

import dlearn.stats as stats
from dlearn.utils.serialize import load_data, save_data


# Program arguments parser
desctxt = """
Train latent network. Use learned attribute and segmentation network.
"""

dataset_txt = """
The input dataset data_name.pkl.
"""

model_txt = """
The model model_name.pkl.
"""

output_txt = """
If not specified, the output stats will be saved as stats.pkl.
Otherwise it will be saved as stats_name.pkl.
"""

parser = argparse.ArgumentParser(description=desctxt)
parser.add_argument('-d', '--dataset', nargs=1, required=True,
                    metavar='name', help=dataset_txt)
parser.add_argument('-m', '--model', nargs=1, required=True,
                    metavar='name', help=model_txt)
parser.add_argument('-o', '--output', nargs='?', default=None,
                    metavar='name', help=output_txt)
parser.add_argument('--display-only', action='store_true')

args = parser.parse_args()


def compute_output(model, subset):
    if isinstance(subset.input, list):
        X, S = subset.input
        f = theano.function(
            inputs=model.input,
            outputs=model.output,
            on_unused_input='ignore'
        )
    else:
        X = subset.input
        f = theano.function(
            inputs=[model.input],
            outputs=model.output,
            on_unused_input='ignore'
        )

    m = X.cpu_data.shape[0]

    batch_size = 100
    n_batches = (m - 1) // batch_size + 1

    output = []
    if isinstance(subset.input, list):
        for i in xrange(n_batches):
            output.append(f(X.cpu_data[i * batch_size: (i + 1) * batch_size],
                            S.cpu_data[i * batch_size: (i + 1) * batch_size]))
    else:
        for i in xrange(n_batches):
            output.append(f(X.cpu_data[i * batch_size: (i + 1) * batch_size]))
    output = np.vstack(output)

    return output


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
        plt.subplot(n_rows, n_cols, j)
        plt.plot(fpr, tpr)
        plt.title('AUC = {:.2f}%'.format(auc * 100))

    plt.show()


if __name__ == '__main__':
    dataset_file = 'data_{0}.pkl'.format(args.dataset[0])
    model_file = 'model_{0}.pkl'.format(args.model[0])
    out_file = 'stats.pkl' if args.output is None else \
               'stats_{0}.pkl'.format(args.output)

    if not args.display_only:
        dataset = load_data(dataset_file)
        model = load_data(model_file)

        output = compute_output(model, dataset.test)
        ret = compute_stats(output, dataset.test.target.cpu_data)

        save_data(ret, out_file)

    ret = load_data(out_file)
    show_stats(ret)
