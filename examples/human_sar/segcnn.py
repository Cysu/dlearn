import os
import sys
import cPickle
import numpy as np
import theano.tensor as T

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

from dlearn.models.layer import FullConnLayer, ConvPoolLayer
from dlearn.models.nnet import NeuralNet
from dlearn.utils import actfuncs, costfuncs
from dlearn.optimization import sgd


def load_data():
    with open('data.pkl', 'rb') as f:
        dataset = cPickle.load(f)

    return dataset


def load_attr_model():
    with open('model_scpool.pkl', 'rb') as f:
        attr_model = cPickle.load(f)

    return attr_model

def scale_per_channel(F):
    fmin, fmax = F.min(axis=[2, 3]), F.max(axis=[2, 3])
    return -1 + 2.0 * (F - fmin.dimshuffle(0, 1, 'x', 'x')) / \
        (fmax - fmin).dimshuffle(0, 1, 'x', 'x')


def train_model(dataset, attr_model):
    X = T.tensor4()
    A = T.matrix()
    S = T.tensor3()

    layers = []
    layers.append(ConvPoolLayer(
        input=X,
        input_shape=(3, 160, 80),
        filter_shape=(32, 3, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False,
        # W=attr_model.blocks[0]._W,
        # b=attr_model.blocks[0]._b,
        # const_params=True
    ))

    layers.append(ConvPoolLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        filter_shape=(64, 32, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False,
        # W=attr_model.blocks[1]._W,
        # b=attr_model.blocks[1]._b,
        # const_params=True
    ))

    layers.append(FullConnLayer(
        # input=scale_per_channel(layers[-1].output).flatten(2),
        input=layers[-1].output.flatten(2),
        input_shape=np.prod(layers[-1].output_shape),
        output_shape=1024,
        dropout_ratio=0.1,
        active_func=actfuncs.tanh
    ))

    layers.append(FullConnLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        output_shape=37 * 17,
        dropout_input=layers[-1].dropout_output,
        active_func=actfuncs.sigmoid
    ))

    model = NeuralNet(layers, [X, A], layers[-1].output)
    model.target = S

    '''
    model.cost = costfuncs.binxent(layers[-1].dropout_output, S.flatten(2)) + \
        1e-3 * model.get_norm(2)
    model.error = costfuncs.binerr(layers[-1].output, S.flatten(2))
    '''

    model.cost = costfuncs.weighted_norm2(layers[-1].dropout_output, S.flatten(2), 1.0) + \
                 1e-3 * model.get_norm(2)
    model.error = costfuncs.weighted_norm2(layers[-1].output, S.flatten(2), 1.0)

    sgd.train(model, dataset, lr=1e-2, momentum=0.9,
              batch_size=100, n_epochs=300,
              epoch_waiting=10)

    return model


def save_model(model):
    with open('model_segcnn.pkl', 'wb') as f:
        cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dataset = load_data()
    attr_model = load_attr_model()
    model = train_model(dataset, attr_model)
    save_model(model)
