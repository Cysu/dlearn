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


def train_model(dataset):
    X = T.tensor4()
    Y = T.matrix()

    edge_layer = ConvPoolLayer(
        input=X,
        input_shape=(3, 80, 30),
        filter_shape=(20, 3, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh
    )

    feat_layer_1 = ConvPoolLayer(
        input=edge_layer.output,
        input_shape=edge_layer.output_shape,
        filter_shape=(50, 20, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=True
    )

    rcon_layer = FullConnLayer(
        input=feat_layer_1.output,
        input_shape=feat_layer_1.output_shape,
        output_shape=np.prod(edge_layer.output_shape[-2:]),
        active_func=actfuncs.tanh
    )

    region = rcon_layer.output
    region = region.reshape((
        region.shape[0],
        edge_layer.output_shape[-2],
        edge_layer.output_shape[-1]
    ))
    region = (region >= 0)

    feat_layer_2 = ConvPoolLayer(
        input=edge_layer.output * region.dimshuffle(0, 'x', 1, 2),
        input_shape=edge_layer.output_shape,
        filter_shape=(50, 20, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=True
    )

    layers = [edge_layer, feat_layer_1, rcon_layer, feat_layer_2]

    layers.append(FullConnLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        output_shape=500,
        active_func=actfuncs.tanh
    ))

    layers.append(FullConnLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        output_shape=99,
        active_func=actfuncs.sigmoid
    ))

    model = NeuralNet(layers, X, layers[-1].output)
    model.target = Y
    model.cost = costfuncs.binxent(layers[-1].output, Y)
    model.error = costfuncs.binerr(layers[-1].output, Y)

    sgd.train(model, dataset, lr=1e-3, momentum=0.9,
              batch_size=500, n_epochs=200,
              lr_decr=1.0)

    return model


def save_model(model):
    with open('model_lnn.pkl', 'wb') as f:
        cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dataset = load_data()
    model = train_model(dataset)
    save_model(model)
