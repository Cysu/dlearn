import os
import sys
import cPickle
import numpy as np
import theano

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

from dlearn.visualization import show_channels


def load_data():
    with open('data.pkl', 'rb') as f:
        dataset = cPickle.load(f)
    return dataset


def load_model():
    with open('model_scpool.pkl', 'rb') as f:
        model = cPickle.load(f)
    return model


def visualize(model, subset, folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    f = theano.function(
        inputs=model.input,
        outputs=model.blocks[0].output,
        on_unused_input='ignore'
    )

    X, __ = subset.input

    __, height, width = X.cpu_data[0].shape
    output_shape = model.blocks[0].output_shape

    fake_S = np.ones((100, height, width), dtype=theano.config.floatX)

    y = f(X.cpu_data[0:100], fake_S)

    for i in xrange(100):
        print 'Saving figure {0}'.format(i)
        show_channels(y[i].reshape(output_shape), n_cols=8,
                      ofpath=os.path.join(folder, '{:04d}.png'.format(i)))


if __name__ == '__main__':
    dataset = load_data()
    model = load_model()
    visualize(model, dataset.test, 'filter_responses')
