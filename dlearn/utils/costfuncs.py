import theano.tensor as T


def binxent(output, target):
    r"""Return the mean binary cross entropy cost.

    The binary cross entropy of two :math:`n`-dimensional vectors :math:`o` and
    :math:`t` is

    .. math::
        c = -\sum_{i=1}^n t_i\log(o_i) + (1-t_i)\log(1-o_i)

    Parameters
    ----------
    output : theano.tensor.matrix
        The output symbol of the model. Each row is a sample vector.
    target : theano.tensor.matrix
        The target symbol of the model. Each row is a ground-truth vector.

    Returns
    -------
    out : theano.tensor.scalar
        The mean of binary cross entropies of all the samples.

    """
    return T.nnet.binary_crossentropy(output, target).sum(axis=1).mean()


def mse(output, target):
    r"""Return the mean square error cost.

    The square error of two :math:`n`-dimensional vectors :math:`o` and
    :math:`t` is

    .. math::
        c = -\sum_{i=1}^n (o_i-t_i)^2

    Parameters
    ----------
    output : theano.tensor.matrix
        The output symbol of the model. Each row is a sample vector.
    target : theano.tensor.matrix
        The target symbol of the model. Each row is a ground-truth vector.

    Returns
    -------
    out : theano.tensor.scalar
        The mean square error of all the samples.

    """

    return ((output - target) ** 2).sum(axis=1).mean()


def neglog(output, target):
    r"""Return the mean negative log-likelihood cost.

    The negative log-likelihood of an output vector :math:`o` with ground-truth
    label :math:`t` is

    .. math::
        c = -\log(o_t)

    Parameters
    ----------
    output : theano.tensor.matrix
        The output symbol of the model. Each row is a sample vector.
    target : theano.tensor.ivector
        The target symbol of the model. Each row is a ground-truth label.

    Returns
    -------
    out : theano.tensor.scalar
        The mean negative log-likelihood of all the samples.

    """
    return -T.mean(T.log(output)[T.arange(target.shape[0]), target])


def miscls_rate(output, target):
    r"""Return the mean misclassification rate.

    Parameters
    ----------
    output : theano.tensor.matrix
        The output symbol of the model. Each row is a sample vector.
    target : theano.tensor.ivector
        The target symbol of the model. Each row is a ground-truth label.

    Returns
    -------
    out : theano.tensor.scalar
        The mean misclassification rate of all the samples.

    """
    pred = T.argmax(output, axis=1)
    return T.neq(pred, target).mean()


def binerr(output, target):
    r"""Return the mean binary prediction error rate.

    The output vector :math:`o\in [0,1]^n`, and target vector :math:`t\in
    \{0,1\}^n`. Then the binary prediction error rate is

    .. math::
        c = \sum_{i=1}^n \delta(round(o_i) = t_i)

    Parameters
    ----------
    output : theano.tensor.matrix
        The output symbol of the model. Each row is a sample vector.
    target : theano.tensor.matrix
        The target symbol of the model. Each row is a ground-truth vector.

    Returns
    -------
    out : theano.tensor.scalar
        The mean binary prediction error rate of all the samples.

    """
    pred = T.round(output)
    return T.neq(pred, target).sum(axis=1).mean()


def KL(target, output):
    r"""Return the mean of summation of Kullback-Leibler divergence.

    **Note that the parameters order is different from other cost functions due
    to the conventional definition of the KL-divergence.** Denote the target
    vector and output vector by :math:`t\in [0,1]^n` and :math:`o\in [0,1]^n`
    respectively, the KL-divergence of each element is defined to be

    .. math::
        KL(t_i||o_i) = t_i\log\frac{t_i}{o_i} + (1-t_i)\log\frac{1-t_i}{1-o_i}

    And the summation over all the elements is

    .. math::
        c = \sum_{i=1}^n KL(t_i||o_i)

    Parameters
    ----------
    target : theano.tensor.matrix
        The target symbol of the model. Each row is a ground-truth vector.
    output : theano.tensor.matrix
        The output symbol of the model. Each row is a sample vector.

    Returns
    -------
    out : theano.tensor.scalar
        The mean of summation of KL-divergence over all the elements.
    """
    kl = target * T.log(target / output) + \
        (1.0 - target) * T.log((1.0 - target) / (1.0 - output))
    return kl.sum(axis=1).mean()


def weighted_norm2(output, target, weight):
    x = (output - target) ** 2
    w = (target + weight)
    return (w * x).sum(axis=1).mean()
