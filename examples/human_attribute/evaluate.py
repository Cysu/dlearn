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
    with open('data.pkl', 'rb') as f:
        dataset = cPickle.load(f)
    return dataset


def load_model():
    with open('model_{0}.pkl'.format(mdname), 'rb') as f:
        model = cPickle.load(f)
    return model


def compute_output(model, subset):
    f = theano.function(
        inputs=model.input,
        outputs=model.output,
        on_unused_input='ignore'
    )

    X, S = subset.input

    m = X.cpu_data.shape[0]

    batch_size = 100
    n_batches = (m - 1) // batch_size + 1

    output = []
    for i in xrange(n_batches):
        output.append(f(X.cpu_data[i * batch_size: (i + 1) * batch_size],
                        S.cpu_data[i * batch_size: (i + 1) * batch_size]))
    output = np.vstack(output)

    return output


def compute_stats(output, target):
    import matplotlib.pyplot as plt
    import matplotlib.figure as fig

    n = target.shape[1]
    n_cols = 4
    n_rows = (n // 4) + 1

    for j in xrange(n):
        # Compute stats
        o = output[:, j].ravel()
        t = target[:, j].ravel()
        fpr, tpr, thresh = stats.roc(o, t)
        auc = stats.auc(fpr, tpr)

        # Plot stats
        plt.subplot(n_rows, n_cols, j)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC = {0}'.format(auc))

    plt.show()
    fig.savefig('stats.png')


if __name__ == '__main__':
    dataset = load_data()
    model = load_model()
    output = compute_output(model, dataset.test)
    compute_stats(output, dataset.test.target.cpu_data)
