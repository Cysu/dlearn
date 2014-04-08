import os
import sys
import cPickle
import theano.tensor as T

if not '..' in sys.path:
    sys.path.insert(0, '..')

from dlearn.data.dataset import Dataset
from dlearn.models.layer import FullConnLayer, ConvPoolLayer
from dlearn.models.nnet import NeuralNet
from dlearn.utils import actfuncs, costfuncs
from dlearn.utils.math import create_shared
from dlearn.optimization import sgd


def load_data():
    fpath = os.path.join('..', 'data', 'mnist', 'mnist.pkl')

    with open(fpath, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    dataset = Dataset()
    dataset.train_x = create_shared(train_set[0])
    dataset.train_y = create_shared(train_set[1], 'int32')

    dataset.valid_x = create_shared(valid_set[0])
    dataset.valid_y = create_shared(valid_set[1], 'int32')

    dataset.test_x = create_shared(test_set[0])
    dataset.test_y = create_shared(test_set[1], 'int32')

    return dataset


def train_model(dataset):
    X = T.matrix()
    Y = T.ivector()

    layers = []
    layers.append(ConvPoolLayer(
        X, (20, 1, 5, 5), (2, 2), (1, 28, 28), actfuncs.tanh))

    layers.append(ConvPoolLayer(
        layers[0].output, (50, 20, 5, 5), (2, 2), None, actfuncs.tanh))

    layers.append(
        FullConnLayer(layers[1].output.flatten(2), 800, 500, actfuncs.tanh))

    layers.append(FullConnLayer(layers[2].output, 500, 10, actfuncs.softmax))

    model = NeuralNet(layers, X, layers[3].output)
    model.target = Y
    model.cost = costfuncs.neglog(layers[3].output, Y)
    model.error = costfuncs.miscls_rate(layers[3].output, Y)

    sgd.train(model, dataset, lr=0.1, momentum=0.9,
              batch_size=500, n_epochs=200,
              lr_decr=1.0)

    return model

if __name__ == '__main__':
    dataset = load_data()
    model = train_model(dataset)
