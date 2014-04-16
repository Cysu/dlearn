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
    Y = T.matrix()

    layers = []
    layers.append(ConvPoolLayer(
        input=X,
        input_shape=(3, 128, 48),
        filter_shape=(20, 3, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.sigmoid
    ))

    layers.append(ConvPoolLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        filter_shape=(50, 20, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.sigmoid,
        flatten=True
    ))

    layers.append(FullConnLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        output_shape=500,
        active_func=actfuncs.sigmoid
    ))

    layers.append(FullConnLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        output_shape=50,
        active_func=actfuncs.sigmoid
    ))

    model = NeuralNet(layers, X, layers[-1].output)
    model.target = Y
    model.cost = costfuncs.binxent(layers[-1].output, Y) + \
        1e-3 * model.get_norm(2) + \
        1e-3 * costfuncs.KL(0.05, layers[1].output)
    model.error = costfuncs.binerr(layers[-1].output, Y)

    sgd.train(model, dataset, lr=1e-3, momentum=0.9,
              batch_size=100, n_epochs=200,
              lr_decr=1.0)

    return model


def save_model(model):
    with open('model_cnn.pkl', 'wb') as f:
        cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dataset = load_data()
    model = train_model(dataset)
    save_model(model)
