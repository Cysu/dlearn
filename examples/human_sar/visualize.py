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
    if not os.isdir(folder):
        os.makedirs(folder)

    f = theano.function(
        inputs=model.input,
        outputs=model.blocks[0].output,
        on_unused_input='ignore'
    )

    X, __ = subset.input

    input_shape = X.cpu_data[0].shape
    output_shape = model.blocks[0].output_shape

    fake_S = np.zeros(input_shape, dtype=theano.config.floatX)[np.newaxis]

    for i in xrange(100):
        y = f(X.cpu_data[i:i + 1], fake_S).reshape(output_shape)
        show_channels(y, n_cols=8, normalize=[-1, 1],
                      ofpath=os.path.join(folder, '{:04d}.png'.format(i)))


if __name__ == '__main__':
    dataset = load_data()
    model = load_model()
    visualize(model, dataset.test, 'filter_response')
