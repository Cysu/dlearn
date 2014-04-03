import theano.tensor as T


def identity(x):
    return x


def relu(x):
    return x * (x > 0.0)


sigmoid = T.nnet.sigmoid

softmax = T.nnet.softmax

tanh = T.tanh
