import os
import sys
import cPickle
import numpy as np
import theano

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

import dlearn.visualization as vis


def load_data():
    with open('data.pkl', 'rb') as f:
        dataset = cPickle.load(f)
    return dataset


def load_model():
    with open('model_filterlc.pkl', 'rb') as f:
        model = cPickle.load(f)
    return model


def visualize(model, subset):
    if not os.path.isdir('mpl_output'):
        os.mkdir('mpl_output')

    f = theano.function(
        inputs=model.input,
        outputs=model.output,
        on_unused_input='ignore'
    )

    X, A = subset.input
    S = subset.target

    for i in xrange(100):
        y = f(X.cpu_data[i:i + 1], A.cpu_data[i:i + 1])
        y = np.vstack((y.reshape(1, 37, 17), S.cpu_data[i:i + 1]))
        fp = os.path.join('mpl_output', '{:04d}.png'.format(i))
        vis.show_channels(y, n_cols=2, normalize=[0, 1], ofpath=fp)


if __name__ == '__main__':
    dataset = load_data()
    model = load_model()
    visualize(model, dataset.test)
