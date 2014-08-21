import time
import numpy as np
import theano
import theano.tensor as T

from ..utils.math import create_empty


def _bind_data(model, subset, irange=None):
    givens = []
    if irange is not None:
        l, r = irange

        if isinstance(model.input, list):
            for sym, val in zip(model.input, subset.input):
                offset = val.cur_irange[0]
                givens.append((sym, val.gpu_data[l - offset:r - offset]))
        else:
            offset = subset.input.cur_irange[0]
            givens.append(
                (model.input, subset.input.gpu_data[l - offset:r - offset]))

        if isinstance(model.target, list):
            for sym, val in zip(model.target, subset.target):
                offset = val.cur_irange[0]
                givens.append((sym, val.gpu_data[l - offset:r - offset]))
        else:
            offset = subset.target.cur_irange[0]
            givens.append(
                (model.target, subset.target.gpu_data[l - offset:r - offset]))
    else:
        for sym, val in zip(model.input, subset.input):
            givens.append((sym, val))
        for sym, val in zip(model.target, subset.target):
            givens.append((sym, val))

    return givens


def train(model, dataset, lr=1e-4, momentum=0.9,
          batch_size=100, n_epochs=100, valid_freq=None,
          patience_incr=2.0, lr_decr=0.5, epoch_waiting=10,
          never_stop=False):
    r"""Train the model with mini-batch Stochastic Gradient Descent(SGD).

    Parameters
    ----------
    model : NeuralNet
        The model to be trained.
    dataset : Dataset
        The dataset to provide training, validation, and testing data.
    lr : float or double, optional
        The initial learning rate. Default is 1e-4.
    momentum : float or double, optional
        The coefficient of momentum term. Default is 0.9.
    batch_size : int, optional
        The number of samples in each mini-batch. Default is 100.
    n_epochs : int, optional
        The number of training epochs. Default is 100.
    valid_freq : None or int, optional
        The number of iterations between validations. If None, then it will be
        set to the size of training batch plus one. Default is None.
    patience_incr : float or double, optional
    lr_decr : float or double, optional
    epoch_waiting : float or double, optional
        `patience` is utilized to stop training when the model converges. It is
        initialized as `n_batches` * 20. After each validation process, if
        `current_valid_error` < `best_valid_error`, then `patience` =
        `current_iter` * `patience_incr`. Otherwise if there is no improvement
        in the last `epoch_waiting` epochs, then `lr` = `lr` * `lr_decr`.
        Default `patience_incr` is 2.0, `lr_decr` is 0.5, and epoch_waiting is
        10.
    never_stop : bool, optional
        If True, then the training process will never stop until user
        interrupts, otherwise it will stop when either reaches `n_epochs` or
        `patience` is consumed.

    """
    n_train = len(dataset.train_ind)
    n_valid = len(dataset.valid_ind)
    n_test = len(dataset.test_ind)

    n_train_batches = n_train // batch_size
    n_valid_batches = n_valid // batch_size
    n_test_batches = n_test // batch_size

    i = T.iscalar()     # batch index
    alpha = T.scalar()  # learning rate
    dummy = T.scalar()  # for parameter updates
    l, r = i * batch_size, (i + 1) * batch_size

    # Comupute updates
    grads = T.grad(model.cost, model.parameters,
                   consider_constant=model.consts)
    incs = [create_empty(p) for p in model.parameters]

    inc_updates = []
    param_updates = []
    for p, g, inc in zip(model.parameters, grads, incs):
        inc_updates.append((inc, momentum * inc - alpha * g))
        param_updates.append((p, p + inc))

    # Build functions
    inc_updates_func = theano.function(
        inputs=[i, alpha], outputs=model.cost, updates=inc_updates,
        givens=_bind_data(model, dataset.train, [l, r]),
        on_unused_input='ignore')

    param_updates_func = theano.function(
        inputs=[dummy], outputs=dummy, updates=param_updates,
        on_unused_input='ignore')

    valid_func = theano.function(
        inputs=[i], outputs=[model.cost, model.error],
        givens=_bind_data(model, dataset.valid, [l, r]),
        on_unused_input='ignore')

    test_func = theano.function(
        inputs=[i], outputs=[model.cost, model.error],
        givens=_bind_data(model, dataset.test, [l, r]),
        on_unused_input='ignore')

    # Start training
    best_valid_error = np.inf
    test_error = np.inf
    patience = n_train_batches * 20
    last_improve_epoch = 0
    if valid_freq is None:
        valid_freq = n_train_batches + 1

    print "Start training ..."
    begin_time = time.clock()

    try:
        for epoch in xrange(n_epochs):
            for j in xrange(n_train_batches):
                cur_iter = epoch * n_train_batches + j

                # training
                dataset.train.prepare([j * batch_size, (j + 1) * batch_size])
                batch_cost = inc_updates_func(j, lr)
                param_updates_func(0)
                print "[train] epoch {0} batch {1}/{2}, iter {3}, cost {4}".format(
                    epoch, j + 1, n_train_batches, cur_iter, batch_cost)

                if (cur_iter + 1) % valid_freq == 0:
                    # validation
                    valid_cost, valid_error = [], []
                    for j in xrange(n_valid_batches):
                        dataset.valid.prepare([j * batch_size, (j + 1) * batch_size])
                        cost, error = valid_func(j)
                        valid_cost.append(cost)
                        valid_error.append(error)
                    valid_cost = np.mean(valid_cost)
                    valid_error = np.mean(valid_error)
                    print "[valid] cost {0}, error {1}".format(valid_cost, valid_error)

                    # testing
                    if valid_error < best_valid_error:
                        best_valid_error = valid_error
                        last_improve_epoch = epoch
                        patience = max(patience, cur_iter * patience_incr)
                        print "Update patience {0}".format(patience)

                        test_cost, test_error = [], []
                        for j in xrange(n_test_batches):
                            dataset.test.prepare([j * batch_size, (j + 1) * batch_size])
                            cost, error = test_func(j)
                            test_cost.append(cost)
                            test_error.append(error)
                        test_cost = np.mean(test_cost)
                        test_error = np.mean(test_error)
                        print "[test] cost {0}, error {1}".format(test_cost, test_error)
                    elif epoch >= last_improve_epoch + epoch_waiting:
                        # lr decreasing
                        lr *= lr_decr
                        last_improve_epoch = epoch
                        print "Update lr {0}".format(lr)

            # early stopping
            if cur_iter > patience and not never_stop:
                break

    except KeyboardInterrupt:
        print "Keyboard interrupt. Stop training"

    print "Training complete, time {0}".format(time.clock() - begin_time)
    print "Best validation error {0}, test error {1}".format(best_valid_error,
                                                             test_error)

def train_dp(model, data_provider, lr=1e-4, momentum=0.9,
             n_epochs=100, valid_freq=None,
             patience_incr=2.0, lr_decr=0.1, epoch_waiting=10,
             never_stop=False):
    r"""Train the model with mini-batch Stochastic Gradient Descent(SGD).

    Parameters
    ----------
    model : NeuralNet
        The model to be trained.
    dataset : Dataset
        The dataset to provide training, validation, and testing data.
    lr : float or double, optional
        The initial learning rate. Default is 1e-4.
    momentum : float or double, optional
        The coefficient of momentum term. Default is 0.9.
    batch_size : int, optional
        The number of samples in each mini-batch. Default is 100.
    n_epochs : int, optional
        The number of training epochs. Default is 100.
    valid_freq : None or int, optional
        The number of iterations between validations. If None, then it will be
        set to the size of training batch plus one. Default is None.
    patience_incr : float or double, optional
    lr_decr : float or double, optional
    epoch_waiting : float or double, optional
        `patience` is utilized to stop training when the model converges. It is
        initialized as `n_batches` * 20. After each validation process, if
        `current_valid_error` < `best_valid_error`, then `patience` =
        `current_iter` * `patience_incr`. Otherwise if there is no improvement
        in the last `epoch_waiting` epochs, then `lr` = `lr` * `lr_decr`.
        Default `patience_incr` is 2.0, `lr_decr` is 0.5, and epoch_waiting is
        10.
    never_stop : bool, optional
        If True, then the training process will never stop until user
        interrupts, otherwise it will stop when either reaches `n_epochs` or
        `patience` is consumed.

    """
    n_train_batches = data_provider.train.n_batches
    n_valid_batches = data_provider.valid.n_batches
    n_test_batches = data_provider.test.n_batches

    alpha = T.scalar()  # learning rate
    dummy = T.scalar()  # for parameter updates

    # Comupute updates
    grads = T.grad(model.cost, model.parameters,
                   consider_constant=model.consts)
    incs = [create_empty(p) for p in model.parameters]

    inc_updates = []
    param_updates = []
    for p, g, inc in zip(model.parameters, grads, incs):
        inc_updates.append((inc, momentum * inc - alpha * g))
        param_updates.append((p, p + inc))

    # Build functions
    inc_updates_func = theano.function(
        inputs=[alpha], outputs=model.cost, updates=inc_updates,
        givens=_bind_data(model, data_provider.train),
        on_unused_input='ignore')

    param_updates_func = theano.function(
        inputs=[dummy], outputs=dummy, updates=param_updates,
        on_unused_input='ignore')

    valid_func = theano.function(
        inputs=[], outputs=[model.cost, model.error],
        givens=_bind_data(model, data_provider.valid),
        on_unused_input='ignore')

    test_func = theano.function(
        inputs=[], outputs=[model.cost, model.error],
        givens=_bind_data(model, data_provider.test),
        on_unused_input='ignore')

    # Start training
    best_valid_cost = np.inf
    last_improve_epoch = 0
    train_costs = [0] * n_train_batches
    if valid_freq is None:
        valid_freq = n_train_batches + 1

    print "Start training ..."
    begin_time = time.clock()

    try:
        for epoch in xrange(n_epochs):
            need_stop = False

            for j in xrange(n_train_batches):
                cur_iter = epoch * n_train_batches + j

                # training
                data_provider.train.prepare(j)
                batch_cost = inc_updates_func(lr)
                param_updates_func(0)
                train_costs[j] = batch_cost
                print "[train] epoch {0} batch {1}/{2}, iter {3}, cost {4}".format(
                    epoch, j + 1, n_train_batches, cur_iter, batch_cost)

                if (cur_iter + 1) % valid_freq == 0:
                    # validation
                    valid_cost, valid_error = [], []
                    for j in xrange(n_valid_batches):
                        data_provider.valid.prepare(j)
                        cost, error = valid_func()
                        valid_cost.append(cost)
                        valid_error.append(error)
                    valid_cost = np.mean(valid_cost)
                    valid_error = np.mean(valid_error)
                    print "[valid] cost {0}, error {1}".format(valid_cost, valid_error)

                    # testing
                    if valid_cost < best_valid_cost:
                        best_valid_cost = valid_cost
                        best_valid_error = valid_error
                        last_improve_epoch = epoch

                        test_cost, test_error = [], []
                        for j in xrange(n_test_batches):
                            data_provider.test.prepare(j)
                            cost, error = test_func()
                            test_cost.append(cost)
                            test_error.append(error)
                        test_cost = np.mean(test_cost)
                        test_error = np.mean(test_error)
                        print "[test] cost {0}, error {1}".format(test_cost, test_error)
                    elif epoch >= last_improve_epoch + epoch_waiting:
                        # check overfitting
                        if np.mean(train_costs) < valid_cost * 0.85:
                            need_stop = True
                            break

                        # lr decreasing
                        if lr >= 1e-5:
                            lr *= lr_decr
                            print "Update lr {0}".format(lr)

                        last_improve_epoch = epoch

            # early stopping
            if not never_stop and need_stop:
                break

    except KeyboardInterrupt:
        print "Keyboard interrupt. Stop training"

    print "Training complete, time {0}".format(time.clock() - begin_time)
    print "Best validation cost {0}, valid error {1}".format(best_valid_cost, best_valid_error)
    print "Test cost {0}, test error {1}".format(test_cost, test_error)

