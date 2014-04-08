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
    X = T.matrix()
    Y = T.matrix()

    layers = []
    layers.append(ConvPoolLayer(
        X, (20, 3, 5, 5), (2, 2), (3, 80, 30), actfuncs.tanh))

    layers.append(ConvPoolLayer(
        layers[0].output, (50, 20, 5, 5), (2, 2), None, actfuncs.tanh))

    layers.append(
        FullConnLayer(layers[1].output.flatten(2), 3400, 500, actfuncs.tanh))

    layers.append(FullConnLayer(layers[2].output, 500, 99, actfuncs.sigmoid))

    model = NeuralNet(layers, X, layers[3].output)
    model.target = Y
    model.cost = costfuncs.binxent(layers[3].output, Y)
    model.error = costfuncs.binerr(layers[3].output, Y)

    sgd.train(model, dataset, lr=1e-3, momentum=0.9,
              batch_size=500, n_epochs=200,
              lr_decr=1.0)

    return model

if __name__ == '__main__':
    dataset = load_data()
    model = train_model(dataset)
