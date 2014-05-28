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
    with open('data_attribute.pkl', 'rb') as f:
        dataset = cPickle.load(f)

    return dataset


def load_attr_model():
    with open('model_scpool.pkl', 'rb') as f:
        attr_model = cPickle.load(f)

    return attr_model


def load_seg_model():
    with open('model_segcnn.pkl', 'rb') as f:
        seg_model = cPickle.load(f)

    return seg_model


def pooling(fmaps):
    s = fmaps.sum(axis=[2, 3])
    Z = abs(actfuncs.tanh(fmaps)).sum(axis=[2, 3])
    return s / Z


def train_model(dataset, attr_model, seg_model):
    X = T.tensor4()
    A = T.matrix()

    feature_layers = []
    feature_layers.append(ConvPoolLayer(
        input=X,
        input_shape=(3, 160, 80),
        filter_shape=(32, 3, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False,
        W=attr_model.blocks[0]._W,
        b=0.0
    ))

    feature_layers.append(ConvPoolLayer(
        input=feature_layers[-1].output,
        input_shape=feature_layers[-1].output_shape,
        filter_shape=(64, 32, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False,
        W=attr_model.blocks[1]._W,
        b=0.0
    ))

    seg_layers = []
    seg_layers.append(FullConnLayer(
        input=feature_layers[-1].output.flatten(2),
        input_shape=np.prod(feature_layers[-1].output_shape),
        output_shape=1024,
        dropout_ratio=0.1,
        active_func=actfuncs.tanh
    ))

    seg_layers.append(FullConnLayer(
        input=seg_layers[-1].output,
        input_shape=seg_layers[-1].output_shape,
        output_shape=37 * 17,
        dropout_input=seg_layers[-1].dropout_output,
        active_func=actfuncs.sigmoid
    ))

    S = seg_layers[-1].output
    S = S * (S >= 0.5)
    S = S.reshape((S.shape[0], 37, 17))
    S = S.dimshuffle(0, 'x', 1, 2)

    S_dropout = seg_layers[-1].dropout_output
    S_dropout = S_dropout * (S_dropout >= 0.5)
    S_dropout = S_dropout.reshape((S_dropout.shape[0], 37, 17))
    S_dropout = S_dropout.dimshuffle(0, 'x', 1, 2)

    attr_layers = []
    attr_layers.append(ConvPoolLayer(
        input=feature_layers[-1].output * S,
        input_shape=feature_layers[-1].output_shape,
        filter_shape=(128, 64, 3, 3),
        pool_shape=(2, 2),
        dropout_input=feature_layers[-1].output * S_dropout,
        active_func=actfuncs.tanh,
        flatten=False,
        W=attr_model.blocks[2]._W,
        b=0.0
    ))

    attr_layers.append(FullConnLayer(
        input=pooling(attr_layers[-1].output),
        input_shape=attr_layers[-1].output_shape[0],
        output_shape=64,
        dropout_input=attr_layers[-1].dropout_output,
        dropout_ratio=0.1,
        active_func=actfuncs.tanh,
        W=attr_model.blocks[3]._W,
        b=attr_model.blocks[3]._b
    ))

    attr_layers.append(FullConnLayer(
        input=attr_layers[-1].output,
        input_shape=attr_layers[-1].output_shape,
        output_shape=11,
        dropout_input=attr_layers[-1].dropout_output,
        active_func=actfuncs.sigmoid,
        W=attr_model.blocks[4]._W,
        b=attr_model.blocks[4]._b
    ))

    model = NeuralNet(feature_layers + seg_layers + attr_layers,
                      X, attr_layers[-1].output)
    model.target = A

    model.cost = costfuncs.binxent(attr_layers[-1].dropout_output, A) + \
        1e-3 * model.get_norm(2)
    model.error = costfuncs.binerr(attr_layers[-1].output, A)

    sgd.train(model, dataset, lr=1e-2, momentum=0.9,
              batch_size=100, n_epochs=300,
              epoch_waiting=10)

    return model


def save_model(model):
    with open('model_latent.pkl', 'wb') as f:
        cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dataset = load_data()
    attr_model = load_attr_model()
    seg_model = load_seg_model()

    model = train_model(dataset, attr_model, seg_model)
    save_model(model)
