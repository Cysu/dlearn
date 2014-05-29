import os
import sys
import cPickle
import numpy as np
import theano

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

import dlearn.stats as stats


if len(sys.argv) != 2:
    sys.exit('{0} method_name'.format(sys.argv[0]))
mdname = sys.argv[1]


def load_data():
    with open('data_attribute.pkl', 'rb') as f:
        dataset = cPickle.load(f)
    return dataset


def load_model():
    with open('model_{0}.pkl'.format(mdname), 'rb') as f:
        model = cPickle.load(f)
    return model


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


def save_stats(ret):
    with open('stats_{0}.pkl'.format(mdname), 'wb') as f:
        cPickle.dump(ret, f, cPickle.HIGHEST_PROTOCOL)


def load_stats():
    with open('stats_{0}.pkl'.format(mdname), 'rb') as f:
        ret = cPickle.load(f)
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
    dataset = load_data()
    model = load_model()
    output = compute_output(model, dataset.test)
    ret = compute_stats(output, dataset.test.target.cpu_data)
    save_stats(ret)
    ret = load_stats()
    show_stats(ret)
