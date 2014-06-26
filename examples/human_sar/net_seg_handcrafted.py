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
import conf_cuhk_sar as conf


def choose_seg(seg, title):
    from dlearn.utils import imgproc
    val = conf.seg_pix[title]
    img = (seg == val).astype(np.float32)
    img = imgproc.resize(img, [37, 17])
    return img.astype(np.float32)


def load_dataset():
    from scipy.io import loadmat
    from dlearn.data.dataset import Dataset

    matdata = loadmat(os.path.join('..', '..', 'data', 'human_sar', 'CUHK_SAR.mat'))
    m, n = matdata['images'].shape
    S = [choose_seg(matdata['segmentations'][i, 0], 'Upper')
         for i in xrange(m)]
    S = np.asarray(S)

    matdata = loadmat('XY_CUHK_SAR.mat')
    X = matdata['X'].T.astype('float32')
    X = X / 100.0
    X = X - X.mean(axis=0)

    dataset = Dataset(X, S)
    dataset.split(0.7, 0.2)

    return dataset


def train_model(dataset):
    X = T.matrix()
    S = T.tensor3()

    layers = []

    layers.append(FullConnLayer(
        input=X,
        input_shape=2784,
        output_shape=1024,
        dropout_ratio=0.1,
        active_func=actfuncs.tanh
    ))

    """
    layers.append(FullConnLayer(
        input=layers[-1].output,
        input_shape=layers[-1].output_shape,
        output_shape=1024,
        dropout_ratio=0.1,
        dropout_input=layers[-1].dropout_output,
        active_func=actfuncs.tanh
    ))
    """

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
              epoch_waiting=10, never_stop=True)

    return model


if __name__ == '__main__':
    dataset = load_dataset()

    model = train_model(dataset)

    save_data(model, 'model_seg_handcrafted.pkl')
