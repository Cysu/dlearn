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
    with open('model_cnn.pkl', 'rb') as f:
        model = cPickle.load(f)
    return model


def visualize(model, dataset):
    f = theano.function(
        inputs=[model.input],
        outputs=model.blocks[0].output
    )

    x = dataset.test_x.get_value(borrow=True)
    for i in xrange(x.shape[0]):
        y = f(x[i:i + 1]).squeeze()
        vis.show_channels(y, n_cols=16)


if __name__ == '__main__':
    dataset = load_data()
    model = load_model()
    visualize(model, dataset)
