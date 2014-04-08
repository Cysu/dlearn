import numpy as np
import theano
import theano.tensor as T

from .block import Block
from ..utils import actfuncs
from ..utils.math import nprng


class FullConnLayer(Block):

    r"""Construct a fully connected layer.

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
    active_func : None, theano.Op or function, optional
        If None, then no active function will be applied to the output,
        otherwise the specified will be applied. Default is None.
    W : None or theano.matrix(shared), optional
        If None, the weight matrix will be intialized randomly, otherwise it
        will be set to the specified value. Default is None.
    b : None or theano.vector(shared), optional
        If None, the bias vector will be initialized as zero, otherwise it will
        be set to the specified value. Default is None.

    """

    def __init__(self, input, input_size, output_size,
                 active_func=None, W=None, b=None):
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


class ConvPoolLayer(Block):

    """Construct a convolutional and max-pooling layer.

    The output of the convolutional and max-pooling layer is

    .. math::
        y = \sigma(\phi(\sum_{i=1}^nW_i*x+b_i))

    where :math:`x` is an input sample vector, :math:`y` is the output vector,
    :math:`W_i` is the filter matrix, :math:`b_i` is the bias vector,
    :math:`\phi` is the max-pooling function and :math:`\sigma` is the active
    function.

    Parameters
    ----------
    input : theano.tensor.matrix or theano.tensor.tensor4
        The input symbol of the fully connected layer. Along the first dimension
        are image samples.
    filter_shape : list/tuple of int
        The shape of the filter matrix. (n_filters, n_channels, n_rows, n_cols).
    pool_shape : list/tuple of int
        The shape of the pooling region. (n_rows, n_cols).
    image_shape : None or list/tuple of int, optional
        If None, then the input should be ``theano.tensor.tensor4``, otherwise
        each input vector will be reshaped to image_shape. (n_channels, n_rows,
        n_cols). Default is None.
    active_func : None, theano.Op or function, optional
        If None, then no active function will be applied to the output,
        otherwise the specified will be applied. Default is None.

    """

    def __init__(self, input, filter_shape, pool_shape,
                 image_shape=None, active_func=None):
        super(ConvPoolLayer, self).__init__(input)

        self._input = input
        self._filter_shape = filter_shape
        self._pool_shape = pool_shape
        self._image_shape = image_shape
        self._active_func = active_func

        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * \
            np.prod(filter_shape[2:]) / np.prod(pool_shape)

        W_bound = np.sqrt(6.0 / (fan_in + fan_out))

        if active_func == actfuncs.sigmoid:
            W_bound *= 4

        init_W = np.asarray(nprng.uniform(low=-W_bound, high=W_bound,
                                          size=filter_shape),
                            dtype=theano.config.floatX)

        self._W = theano.shared(value=init_W, borrow=True)

        init_b = np.zeros((filter_shape[0],), dtype=theano.config.floatX)

        self._b = theano.shared(value=init_b, borrow=True)

        self._params = [self._W, self._b]

        x = self._input if self._image_shape is None else \
            self._input.reshape((self._input.shape[0],) + self._image_shape)

        z = T.nnet.conv.conv2d(input=x, filters=self._W,
                               filter_shape=self._filter_shape)

        z = T.signal.downsample.max_pool_2d(input=z, ds=self._pool_shape,
                                            ignore_border=True)

        z = z + self._b.dimshuffle('x', 0, 'x', 'x')

        self._output = z if self._active_func is None else self._active_func(z)

    def get_norm(self, l):
        """Return the norm of the filter matrix.

        Parameters
        ----------
        l : int
            The L?-norm.

        Returns
        -------
        out : theano.tensor.scalar
            The norm of the filter matrix.

        """
        return self._W.norm(l)
