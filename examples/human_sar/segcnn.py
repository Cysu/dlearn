import os
import sys
import cPickle
import theano.tensor as T

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

from dlearn.models.layer import FullConnLayer, ConvPoolLayer, LinearCombLayer
from dlearn.models.nnet import NeuralNet
from dlearn.models.autoencoder import AutoEncoder
from dlearn.utils import actfuncs, costfuncs
from dlearn.optimization import sgd


def load_data():
    with open('data.pkl', 'rb') as f:
        dataset = cPickle.load(f)

    return dataset


def train_model(dataset):
    X = T.tensor4()
    S = T.tensor3()

    layers = []
    layers.append(ConvPoolLayer(
        input=X,
        input_shape=(3, 160, 80),
        filter_shape=(32, 3, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False,
        b=0.0
    ))

    layers.append(ConvPoolLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        filter_shape=(64, 32, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False,
        b=0.0
    ))

    layers.append(LinearCombLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        dropout_ratio=0.1,
        active_func=actfuncs.sigmoid,
        flatten=True
    ))

    ae = AutoEncoder(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        hidden_shape=[256, 128],
        dropout_input=layers[-1].dropout_output,
        dropout_ratio=0.1,
        active_func=actfuncs.sigmoid
    )

    model = NeuralNet(layers + [ae], X, ae.output)
    model.target = S
    model.cost = costfuncs.binxent(ae.dropout_output, S.flatten(2)) + \
        1e-3 * model.get_norm(2)
    model.error = costfuncs.binerr(ae.output, S.flatten(2))

    sgd.train(model, dataset, lr=1e-3, momentum=0.9,
              batch_size=100, n_epochs=300,
              epoch_waiting=10)

    return model


def save_model(model):
    with open('model_segcnn.pkl', 'wb') as f:
        cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dataset = load_data()
    model = train_model(dataset)
    save_model(model)
