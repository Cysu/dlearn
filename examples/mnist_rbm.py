import os
import sys
import cPickle
import time
import numpy as np
import theano
import theano.tensor as T

homepath = '..'
if not homepath in sys.path:
    sys.path.insert(0, homepath)

from dlearn.data.dataset import Dataset
from dlearn.models.rbm import RBM
from dlearn.utils import Wrapper
from dlearn.utils.math import nprng
from dlearn.visualization import show_channels


def load_data():
    fpath = os.path.join('..', 'data', 'mnist', 'mnist.pkl')

    with open(fpath, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    dataset = Dataset(
        train=Wrapper(
            input=train_set[0],
            target=train_set[1]
        ),
        valid=Wrapper(
            input=valid_set[0],
            target=valid_set[1]
        ),
        test=Wrapper(
            input=test_set[0],
            target=test_set[1]
        ),
        limit=None
    )

    return dataset


def train_model(dataset):
    # Construct the RBM model
    X = T.matrix()
    index = T.lscalar()

    n_epochs = 15
    batch_size = 20
    vshape, hshape = 784, 500
    n_batches = dataset.train.input.cpu_data.shape[0] / batch_size

    persistent_chain = theano.shared(np.zeros((batch_size, hshape),
                                              dtype=theano.config.floatX),
                                     borrow=True)

    rbm = RBM(X, vshape, hshape)

    cost, updates = rbm.get_cost_updates(lr=0.1,
                                         persistent=persistent_chain,
                                         k=15)

    # Train the model
    if not os.path.isdir('rbm_plots'):
        os.mkdir('rbm_plots')

    train_data = dataset.train.input.cpu_data.astype(theano.config.floatX)
    train_data = theano.shared(train_data, borrow=True)

    train_rbm = theano.function(
        inputs=[index], outputs=cost, updates=updates,
        givens={X: train_data[index * batch_size: (index + 1) * batch_size]})

    plot_time = 0
    start_time = time.clock()

    for epoch in xrange(n_epochs):
        mean_cost = np.mean([train_rbm(i) for i in xrange(n_batches)])

        print 'Training epoch {0}, cost {1}'.format(epoch, mean_cost)

        plot_start = time.clock()
        W = rbm._W.get_value(borrow=True).T[0:100]
        ofpath = os.path.join(
            'rbm_plots', 'filter_epoch_{:03d}.png'.format(epoch))
        show_channels(W.reshape((100, 28, 28)),
                      n_cols=10, grayscale=True, ofpath=ofpath)
        plot_stop = time.clock()
        plot_time += plot_stop - plot_start

    stop_time = time.clock()

    print 'Training finished, time {0}'.format(
        stop_time - start_time - plot_time)


def save_model(model):
    with open('model.pkl', 'wb') as f:
        cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)


def gen_samples(dataset, model):
    n_samples = 10
    n_chains = 20
    plot_every = 1000

    test_idx = nprng.randint(dataset.test.input.cpu_data.shape[0] - n_chains)

    persistent_chain = theano.shared(
        dataset.test.input.cpu_data[test_idx: test_idx + n_chains].astype(
            theano.config.floatX),
        borrow=True)

    [pre_sigmoid_hs, h_means, h_samples,
     pre_sigmoid_vs, v_means, v_samples], updates = \
        theano.scan(model.gibbs_vhv,
                    outputs_info=[None] * 5 + [persistent_chain],
                    n_step=plot_every)

    updates.update({persistent_chain: v_samples[-1]})

    sample_fn = theano.function(
        inputs=[], outputs=v_samples[-1], updates=updates)

    evolutions = np.vstack([sample_fn() for __ in xrange(n_samples)])
    evolutions = evolutions.reshape((n_samples * n_chaines, 28, 28))

    show_channels(evolutions, n_cols=n_chains, grayscale=True,
                  ofpath=os.path.join('rbm_plots', 'samples.png'))


if __name__ == '__main__':
    dataset = load_data()
    model = train_model(dataset)
    save_model(model)
    gen_samples(dataset, model)
