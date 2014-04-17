import time
import numpy as np
import theano
import theano.tensor as T

from ..utils.math import create_empty


def train(model, dataset, lr=1e-4, momentum=0.9,
          batch_size=100, n_epochs=100,
          patience_incr=2.0, lr_decr=0.5, epoch_waiting=5,
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
    patience_incr, lr_decr : float or double, optional
        `patience` is utilized to stop training when the model converges. It is
        initialized as `n_batches` * 20. After each validation process, if
        `current_valid_error` < `best_valid_error`, then `patience` =
        `current_iter` * `patience_incr`. Otherwise if there is no improvement
        in the last `epoch_waiting` epochs, then `lr` = `lr` * `lr_decr`.
        Default `patience_incr` is 2.0, and `lr_decr` is 0.5.
    never_stop : bool, optional
        If True, then the training process will never stop until user
        interrupts, otherwise it will stop when either reaches `n_epochs` or
        `patience` is consumed.

    """
    n_train = dataset.train_x.get_value(borrow=True).shape[0]
    n_valid = dataset.valid_x.get_value(borrow=True).shape[0]
    n_test = dataset.test_x.get_value(borrow=True).shape[0]

    n_train_batches = n_train // batch_size
    n_valid_batches = n_valid // batch_size
    n_test_batches = n_test // batch_size

    i = T.iscalar()  # mini-batch index
    alpha = T.scalar()  # learning rate
    dummy = T.scalar()  # for parameter updates

    # Comupute updates
    grads = T.grad(model.cost, model.parameters)

    incs = [create_empty(p) for p in model.parameters]

    inc_updates = []
    param_updates = []

    for p, g, inc in zip(model.parameters, grads, incs):
        inc_updates.append((inc, momentum * inc - alpha * g))
        param_updates.append((p, p + inc))

    # Build functions
    inc_updates_func = theano.function(
        inputs=[i, alpha], outputs=model.cost, updates=inc_updates,
        givens={
            model.input: dataset.train_x[i * batch_size: (i + 1) * batch_size],
            model.target: dataset.train_y[i * batch_size: (i + 1) * batch_size]
        })

    param_updates_func = theano.function(
        inputs=[dummy], outputs=dummy, updates=param_updates)

    valid_func = theano.function(
        inputs=[i], outputs=model.error,
        givens={
            model.input: dataset.valid_x[i * batch_size: (i + 1) * batch_size],
            model.target: dataset.valid_y[i * batch_size: (i + 1) * batch_size]
        })

    test_func = theano.function(
        inputs=[i], outputs=model.error,
        givens={
            model.input: dataset.test_x[i * batch_size: (i + 1) * batch_size],
            model.target: dataset.test_y[i * batch_size: (i + 1) * batch_size]
        })

    # Start training
    best_valid_error = np.inf
    test_error = np.inf

    patience = n_train_batches * 20

    last_improve_epoch = 0

    print "Start training ..."

    begin_time = time.clock()

    for epoch in xrange(n_epochs):
        print "epoch {0}".format(epoch)

        try:
            # training
            for j in xrange(n_train_batches):
                cur_iter = epoch * n_train_batches + j

                batch_cost = inc_updates_func(j, lr)
                param_updates_func(0)

                print "[train] batch {0}/{1}, iter {2}, cost {3}".format(
                    j + 1, n_train_batches, cur_iter, batch_cost)

            # validation
            valid_error = np.mean([valid_func(j)
                                  for j in xrange(n_valid_batches)])

            print "[valid] error {0}".format(valid_error)

            # testing
            if valid_error < best_valid_error:
                last_improve_epoch = epoch
                patience = max(patience, cur_iter * patience_incr)
                print "Update patience {0}".format(patience)

                best_valid_error = valid_error

                test_error = np.mean([test_func(j)
                                     for j in xrange(n_test_batches)])

                print "[test] error {0}".format(test_error)
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
            break

    print "Training complete, time {0}".format(time.clock() - begin_time)
    print "Best validation error {0}, test error {1}".format(best_valid_error,
                                                             test_error)
