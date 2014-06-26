import os
import sys
import argparse
import numpy as np
import theano.tensor as T

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

from dlearn.models.layer import FullConnLayer, ConvPoolLayer
from dlearn.models.nnet import NeuralNet
from dlearn.utils import actfuncs, costfuncs
from dlearn.utils.serialize import load_data, save_data
from dlearn.optimization import sgd


# Program arguments parser
desctxt = """
Train segmentation network. Use filters learned from attribute network.
"""

dataset_txt = """
The input dataset data_name.pkl.
"""

attr_txt = """
The attribute network model_name.pkl.
"""

output_txt = """
If not specified, the output model will be saved as model_seg.pkl.
Otherwise it will be saved as model_seg_name.pkl.
"""

parser = argparse.ArgumentParser(description=desctxt)
parser.add_argument('-d', '--dataset', nargs=1, required=True,
                    metavar='name', help=dataset_txt)
parser.add_argument('-o', '--output', nargs='?', default=None,
                    metavar='name', help=output_txt)

args = parser.parse_args()


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
        flatten=False
    ))

    layers.append(ConvPoolLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        filter_shape=(64, 32, 5, 5),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False
    ))

    layers.append(FullConnLayer(
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

    model = NeuralNet(layers, X, layers[-1].output)
    model.target = S

    '''
    model.cost = costfuncs.binxent(layers[-1].dropout_output, S.flatten(2)) + \
        1e-3 * model.get_norm(2)
    model.error = costfuncs.binerr(layers[-1].output, S.flatten(2))
    '''

    model.cost = costfuncs.weighted_norm2(
        layers[-1].dropout_output, S.flatten(2), 1.0) + \
        1e-3 * model.get_norm(2)
    model.error = costfuncs.weighted_norm2(
        layers[-1].output, S.flatten(2), 1.0)

    sgd.train(model, dataset, lr=1e-2, momentum=0.9,
              batch_size=100, n_epochs=300,
              epoch_waiting=10)

    return model


if __name__ == '__main__':
    dataset_file = 'data_{0}.pkl'.format(args.dataset[0])
    out_file = 'model_seg.pkl' if args.output is None else \
               'model_seg_{0}.pkl'.format(args.output)

    dataset = load_data(dataset_file)

    model = train_model(dataset)

    save_data(model, out_file)
