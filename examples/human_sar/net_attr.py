import os
import sys
import argparse
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
Train attribute network with segmentation as ground truth. Use shape constrained
pooling.
"""

dataset_txt = """
The input dataset data_name.pkl.
"""

output_txt = """
If not specified, the output model will be saved as model_attr.pkl.
Otherwise it will be saved as model_attr_name.pkl.
"""

parser = argparse.ArgumentParser(description=desctxt)
parser.add_argument('-d', '--dataset', nargs=1, required=True,
                    metavar='name', help=dataset_txt)
parser.add_argument('-o', '--output', nargs='?', default=None,
                    metavar='name', help=output_txt)
parser.add_argument('--no-scpool', action='store_true')

args = parser.parse_args()


def train_model(dataset, use_scpool):

    """
    def shape_constrained_pooling(fmaps):
        beta = 100.0
        s = fmaps.sum(axis=[2, 3])
        # Z = abs(actfuncs.tanh(beta * fmaps)).sum(axis=[2, 3])
        Z = T.neq(fmaps, 0).sum(axis=[2, 3])
        return s / Z
    """
    def shape_constrained_pooling(F, S):
        masked_F = T.switch(T.eq(S, 1), F, 0)
        #masked_F = F * S
        s = masked_F.sum(axis=[2, 3])
        Z = T.neq(masked_F, 0).sum(axis=[2, 3])
        return s / Z

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

    """
    layers.append(ConvPoolLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        filter_shape=(128, 64, 3, 3),
        pool_shape=(2, 2),
        active_func=actfuncs.tanh,
        flatten=False,
        b=0.0
    ))
    """

    """
    F = layers[-1].output * S.dimshuffle(0, 'x', 1, 2)
    F = shape_constrained_pooling(F) if use_scpool else \
        F.flatten(2)
    """
    F = layers[-1].output
    F = shape_constrained_pooling(F, S.dimshuffle(0, 'x', 1, 2)) if use_scpool else F.flatten(2)

    layers.append(FullConnLayer(
        input=F,
        input_shape=layers[-1].output_shape[0],
        output_shape=32,
        dropout_ratio=0.1,
        active_func=actfuncs.tanh
    ))

    layers.append(FullConnLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        output_shape=11,
        dropout_input=layers[-1].dropout_output,
        active_func=actfuncs.sigmoid
    ))

    model = NeuralNet(layers, [X, S], layers[-1].output)
    model.target = A
    model.cost = costfuncs.binxent(layers[-1].dropout_output, A) + \
        1e-3 * model.get_norm(2)
    model.error = costfuncs.binerr(layers[-1].output, A)

    sgd.train(model, dataset, lr=1e-2, momentum=0.9,
              batch_size=100, n_epochs=300,
              epoch_waiting=10)

    return model


if __name__ == '__main__':
    dataset_file = 'data_{0}.pkl'.format(args.dataset[0])
    out_file = 'model_attr.pkl' if args.output is None else \
               'model_attr_{0}.pkl'.format(args.output)

    dataset = load_data(dataset_file)

    model = train_model(dataset, not args.no_scpool)

    save_data(model, out_file)
