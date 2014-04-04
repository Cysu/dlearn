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
