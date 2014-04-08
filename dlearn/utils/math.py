import numpy as np
import theano


nprng = np.random.RandomState(999907)


def create_shared(x, datatype='float32'):
    return theano.shared(np.asarray(x, dtype=datatype), borrow=True)


def create_empty(p):
    return theano.shared(np.zeros_like(p.get_value(borrow=True)), borrow=True)
