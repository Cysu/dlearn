import os
import sys
import cPickle
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
    with open('model_scpool.pkl', 'rb') as f:
        model = cPickle.load(f)
    return model


def visualize(model, subset):
    W = model.blocks[3]._W.get_value(borrow=True)[:, 0]
    W = W.reshape((128, 17, 7))
    vis.show_channels(W, n_cols=32)

    f = theano.function(
        inputs=model.input,
        outputs=model.blocks[2].output,
        on_unused_input='ignore'
    )

    subset.prepare([0, 100])
    X, S = subset.input

    for i in xrange(100):
        y = f(X.cpu_data[i:i + 1], S.cpu_data[i:i + 1])
        # y = y.reshape(model.blocks[2].output_shape)
        y = y.reshape((128, 17, 7))
        vis.show_channels(y, n_cols=32, normalize=[-1, 1])


if __name__ == '__main__':
    dataset = load_data()
    model = load_model()
    visualize(model, dataset.train)
