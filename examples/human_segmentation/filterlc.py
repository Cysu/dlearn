import os
import sys
import cPickle
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
    A = T.matrix()
    S = T.tensor3()

    filter_layers = []
    filter_layers.append(ConvPoolLayer(
        input=X,
        input_shape=(3, 160, 80),
        filter_shape=(32, 3, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False
    ))

    weight_layers = []
    weight_layers.append(FullConnLayer(
        input=A,
        input_shape=11,
        output_shape=64,
        active_func=actfuncs.tanh
    ))

    weight_layers.append(FullConnLayer(
        input=weight_layers[-1].output,
        input_shape=weight_layers[-1].output_shape,
        output_shape=filter_layers[-1].output_shape[0],
        active_func=actfuncs.tanh
    ))

    F = filter_layers[-1].output
    w = weight_layers[-1].output
    wF = (w.dimshuffle(0, 1, 'x', 'x') * F).sum(axis=1)
    wF = actfuncs.sigmoid(wF)

    model = NeuralNet(filter_layers + weight_layers, [X, A], wF)
    model.target = S
    model.cost = costfuncs.binxent(wF.flatten(2), S.flatten(2)) + \
        1e-3 * model.get_norm(2)
    model.error = costfuncs.binerr(wF.flatten(2), S.flatten(2))

    sgd.train(model, dataset, lr=1e-3, momentum=0.9,
              batch_size=100, n_epochs=300,
              epoch_waiting=10)

    return model


def save_model(model):
    with open('model_filterlc.pkl', 'wb') as f:
        cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dataset = load_data()
    model = train_model(dataset)
    save_model(model)
