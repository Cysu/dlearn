import numpy as np
import theano
import theano.tensor as T

from .block import Block
from ..utils import actfuncs
from ..utils.math import nprng


class AutoEncoder(Block):

    def __init__(self, input, input_shape, hidden_shape,
                 dropout_input=None, active_func=None):
        super(AutoEncoder, self).__init__(input, dropout_input)

        self._input_shape = input_shape
        self._hidden_shape = hidden_shape
        self._active_func = active_func

        self._W = []
        self._b = []
        self._c = []
        for i in xrange(len(self._hidden_shape)):
            fan_in = input_shape if i == 0 else hidden_shape[i - 1]
            fan_out = hidden_shape[i]

            W_bound = np.sqrt(6.0 / (fan_in + fan_out))

            if active_func == actfuncs.sigmoid:
                W_bound *= 4

            init_W = np.asarray(
                nprng.uniform(low=-W_bound, high=W_bound,
                              size=(fan_in, fan_out)),
                dtype=theano.config.floatX)

            init_b = np.zeros(fan_out, dtype=theano.config.floatX)

            init_c = np.zeros(fan_in, dtype=theano.config.floatX)

            self._W.append(theano.shared(value=init_W, borrow=True))
            self._b.append(theano.shared(value=init_b, borrow=True))
            self._c.append(theano.shared(value=init_c, borrow=True))

        self._params = self._W + self._b + self._c

        def f(x):
            z = x
            for W, b in zip(self._W, self._b):
                z = T.dot(z, W) + b
                if self._active_func is not None:
                    z = self._active_func(z)

            y = z
            for W, c in reversed(zip(self._W, self._c)):
                y = T.dot(y, W.T) + c
                if self._active_func is not None:
                    y = self._active_func(y)

            return (z, y)

        self._hidden_output, self._output = f(self._input)
        self._dropout_hidden_output, self._dropout_output = \
            f(self._dropout_input)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._input_shape

    @property
    def hidden_output_shape(self):
        return self._hidden_shape[-1]

    @property
    def hidden_output(self):
        return self._hidden_output

    @property
    def dropout_hidden_output(self):
        return self._dropout_hidden_output

    def get_norm(self, l):
        """Return the norm of the weight matrices.

        Parameters
        ----------
        l : int
            The L?-norm.

        Returns
        -------
        out : theano.tensor.scalar
            The norm of the weight matrices.

        """
        return sum([W.norm(l) for W in self._W])
