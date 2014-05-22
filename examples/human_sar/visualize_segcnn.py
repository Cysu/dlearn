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
    with open('model_segcnn.pkl', 'rb') as f:
        model = cPickle.load(f)
    return model


def visualize(model, subset, folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    f = theano.function(
        inputs=[model.input],
        outputs=model.output,
        on_unused_input='ignore'
    )

    X = subset.input
    S = subset.target

    output_shape = (1,) + S.cpu_data[0].shape

    y = f(X.cpu_data[0:100])

    for i in xrange(100):
        print 'Saving figure {0}'.format(i)
        v = np.vstack((y[i].reshape(output_shape), S.cpu_data[i:i+1]))
        show_channels(v, n_cols=2, normalize=[0, 1],
                      ofpath=os.path.join(folder, '{:04d}.png'.format(i)))


if __name__ == '__main__':
    dataset = load_data()
    model = load_model()
    visualize(model, dataset.test, 'segcnn_result')
