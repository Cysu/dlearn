import numpy as np
import theano
import theano.tensor as T


nprng = np.random.RandomState(999907)

tnrng = T.shared_randomstreams.RandomStreams(999907)


def create_shared(x):
    return theano.shared(np.asarray(x), borrow=True)


def create_empty(p):
    return theano.shared(np.zeros_like(p.get_value(borrow=True)), borrow=True)


def dropout(x, ratio):
    mask = tnrng.binomial(n=1, p=1 - ratio, size=x.shape)
    return x * T.cast(mask, x.dtype)
