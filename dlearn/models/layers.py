import numpy as np
import theano
import theano.tensor as T

from .block import Block
from ..utils import actfuncs
from ..utils.math import nprng


class FullConnLayer(Block):

    """Construct a fully connected layer.

    The output of the fully connected layer is
    
    .. math::
        y = \sigma(Wx+b)

    where :math:`x` is an input sample vector, :math:`y` is the output vector,
    :math:`W` is the weight matrix, :math:`b` is the bias vector and
    :math:`\sigma` is the active function.

    Parameters
    ----------
    input : theano.tensor.matrix
        The input symbol of the fully connected layer. Each row is a sample
        vector.
    input_size : int
        The size of each input sample.
    output_size : int
        The size of each output sample.
    W : None or theano.matrix(shared)
        If None, the weight matrix will be intialized randomly, otherwise it
        will be set to the specified value. Default is None.
    b : None or theano.vector(shared)
        If None, the bias vector will be initialized as zero, otherwise it will
        be set to the specified value. Default is None.
    active_func : None, theano.Op or function, optional
        If None, then no active function will be applied to the output,
        otherwise the specified will be applied. Default is None.

    """

    def __init__(self, input, input_size, output_size,
                 W=None, b=None, active_func=None):
        super(FullConnLayer, self).__init__(input)

        self._active_func = active_func

        if W is None:
            W_bound = np.sqrt(6.0 / (input_size + output_size))

            if active_func == actfuncs.sigmoid:
                W_bound *= 4

            init_W = np.asarray(nprng.uniform(low=-W_bound, high=W_bound,
                                              size=(input_size, output_size)),
                                dtype=theano.config.floatX)

            self._W = theano.shared(value=init_W, borrow=True)
        else:
            self._W = W

        if b is None:
            init_b = np.zeros(output_size, dtype=theano.config.floatX)

            self._b = theano.shared(value=init_b, borrow=True)
        else:
            self._b = b

        self._params = [self._W, self._b]

        z = T.dot(self._input, self._W) + self._b
        self._output = z if self._active_func is None else self._active_func(z)

    def get_norm(self, l):
        """Return the norm of the weight matrix.

        Parameters
        ----------
        l : int
            The L?-norm.

        Returns
        -------
        out : theano.tensor.scalar
            The norm of the weight matrix.

        """
        return self._W.norm(l)
