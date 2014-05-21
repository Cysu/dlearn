import os
import sys
import cPickle
import numpy as np
import theano

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
    Y = T.ivector()

    edge_layer_1 = ConvPoolLayer(
        input=X,
        input_shape=(3, 128, 48),
        filter_shape=(32, 3, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh
    )

    feat_layer_1 = ConvPoolLayer(
        input=edge_layer_1.output,
        input_shape=edge_layer_1.output_shape,
        filter_shape=(64, 32, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False
    )

    feat_layer_2 = ConvPoolLayer(
        input=feat_layer_1.output,
        input_shape=feat_layer_1.output_shape,
        filter_shape=(128, 64, 3, 3),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=True
    )

    rcon_layer_1 = FullConnLayer(
        input=feat_layer_2.output,
        input_shape=feat_layer_2.output_shape,
        output_shape=512,
        dropout_ratio=0.1,
        active_func=actfuncs.tanh
    )

    rcon_layer_2 = FullConnLayer(
        input=rcon_layer_1.output,
        input_shape=rcon_layer_1.output_shape,
        output_shape=39,
        dropout_input=rcon_layer_1.dropout_output,
        active_func=actfuncs.sigmoid
    )

    region = rcon_layer_2.output
    region = region.reshape((
        region.shape[0],
        13,
        3
    ))

    region_dropout = rcon_layer_2.dropout_output
    region_dropout = region_dropout.reshape((
        region_dropout.shape[0],
        13,
        3
    ))

    h, w = region_dropout.shape[1], region_dropout.shape[2]
    rim = T.cast(region_dropout, theano.config.floatX)
    dx = (rim[:, :, 0:w - 1] - rim[:, :, 1:w]) ** 2
    dy = (rim[:, 0:h - 1, :] - rim[:, 1:h, :]) ** 2
    smooth_cost = dx.mean(axis=0).sum() + dy.mean(axis=0).sum()

    edge_layer_2 = ConvPoolLayer(
        input=X,
        input_shape=(3, 128, 48),
        filter_shape=(32, 3, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh
    )

    feat_layer_3 = ConvPoolLayer(
        input=edge_layer.output,
        input_shape=edge_layer.output_shape,
        filter_shape=(64, 32, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False
    )

    feat_layer_4 = ConvPoolLayer(
        input=feat_layer_3.output,
        input_shape=feat_layer_3.output_shape,
        filter_shape=(128, 64, 3, 3),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False
    )

    layers = [edge_layer, feat_layer_1, feat_layer_2,
              rcon_layer_1, rcon_layer_2,
              feat_layer_3, feat_layer_4]

    layers.append(FullConnLayer(
        input=(layers[-1].output * region.dimshuffle(0, 'x', 1, 2)).flatten(2),
        input_shape=np.prod(layers[-1].output_shape),
        output_shape=512,
        dropout_ratio=0.1,
        dropout_input=(layers[-1].output *
                       region_dropout.dimshuffle(0, 'x', 1, 2)).flatten(2),
        active_func=actfuncs.tanh
    ))

    layers.append(FullConnLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        output_shape=9,
        dropout_input=layers[-1].dropout_output,
        active_func=actfuncs.softmax
    ))

    model = NeuralNet(layers, X, layers[-1].output)
    model.target = Y
    model.cost = costfuncs.neglog(layers[-1].dropout_output, Y) + \
        1e-3 * model.get_norm(2) + 1e-2 * smooth_cost
    model.error = costfuncs.miscls_rate(layers[-1].output, Y)

    sgd.train(model, dataset, lr=1e-3, momentum=0.9,
              batch_size=100, n_epochs=1000,
              epoch_waiting=20)

    return model


def save_model(model):
    with open('model_lnn.pkl', 'wb') as f:
        cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dataset = load_data()
    model = train_model(dataset)
    save_model(model)
