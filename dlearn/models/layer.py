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

    Attributes
    ----------
    input_shape : int or list/tuple of int
        The shape of each input sample. If the type of `input_shape` is int,
        then each input sample is a vector. *Read-only*.
    output_shape : int or list/tuple of int
        The shape of each output sample. If the type of `output_shape` is int,
        then each output sample is a vector. *Read-only*.

    Parameters
    ----------
    input : theano.tensor.matrix
        The input symbol of the fully connected layer. Each row is a sample
        vector.
    input_shape : int or list/tuple of int
        The shape of each input sample. If the type of `input_shape` is int,
        then each input sample is a vector.
    output_shape : int or list/tuple of int
        The shape of each output sample. If the type of `output_shape` is int,
        then each output sample is a vector.
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

    def __init__(self, input, input_shape, output_shape,
                 active_func=None, W=None, b=None):
        super(FullConnLayer, self).__init__(input)

        if isinstance(input_shape, int):
            self._input_shape = input_shape
        elif isinstance(input_shape, tuple) or isinstance(input_shape, list):
            self._input_shape = np.prod(input_shape)
        else:
            raise ValueError("input_shape type error")

        if isinstance(output_shape, int):
            self._output_shape = output_shape
        elif isinstance(output_shape, tuple) or isinstance(output_shape, list):
            self._output_shape = np.prod(output_shape)
        else:
            raise ValueError("output_shape type error")

        self._active_func = active_func

        if W is None:
            W_bound = np.sqrt(6.0 / (self._input_shape + self._output_shape))

            if active_func == actfuncs.sigmoid:
                W_bound *= 4

            init_W = np.asarray(
                nprng.uniform(low=-W_bound, high=W_bound,
                              size=(self._input_shape, self._output_shape)),
                dtype=theano.config.floatX)

            self._W = theano.shared(value=init_W, borrow=True)
        else:
            self._W = W

        if b is None:
            init_b = np.zeros(self._output_shape, dtype=theano.config.floatX)

            self._b = theano.shared(value=init_b, borrow=True)
        else:
            self._b = b

        self._params = [self._W, self._b]

        z = T.dot(self._input, self._W) + self._b
        self._output = z if self._active_func is None else self._active_func(z)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

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

    Attributes
    ----------
    input_shape : list/tuple of int
        The shape of each input sample. (n_channels, n_rows, n_cols). *Read-
        only*.
    output_shape : int or list/tuple of int
        The shape of each output sample. If the type of `output_shape` is int,
        then each output sample is a vector. *Read-only*.

    Parameters
    ----------
    input : theano.tensor.tensor4
        The input symbol of the fully connected layer. Along the first dimension
        are image samples.
    input_shape : list/tuple of int
        The shape of each input sample. (n_channels, n_rows, n_cols).
    filter_shape : list/tuple of int
        The shape of the filters. (n_filters, n_channels, n_rows, n_cols).
    pool_shape : None or list/tuple of int, optional
        If None, then no max-pooling will be performed, otherwise the shape of
        the max-pooling region should be (n_rows, n_cols). Default if None.
    active_func : None, theano.Op or function, optional
        If None, then no active function will be applied to the output,
        otherwise the specified will be applied. Default is None.
    flatten : bool, optional
        If False, then output will not be flattened, otherwise the output of
        each sample will be flattened into a vector. Default is False.

    """

    def __init__(self, input, input_shape, filter_shape,
                 pool_shape=None, active_func=None, flatten=False):
        super(ConvPoolLayer, self).__init__(input)

        self._input = input
        self._input_shape = input_shape
        self._filter_shape = filter_shape
        self._pool_shape = pool_shape
        self._active_func = active_func
        self._flatten = flatten
        self._output_shape = (
            filter_shape[0],
            (input_shape[-2] - filter_shape[-2] + 1) // pool_shape[0],
            (input_shape[-1] - filter_shape[-1] + 1) // pool_shape[1]
        )
        if flatten:
            self._output_shape = np.prod(self._output_shape)

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

        z = T.nnet.conv.conv2d(input=self._input, filters=self._W,
                               filter_shape=self._filter_shape)

        if self._pool_shape is not None:
            z = T.signal.downsample.max_pool_2d(input=z, ds=self._pool_shape,
                                                ignore_border=True)

        z = z + self._b.dimshuffle('x', 0, 'x', 'x')

        if self._active_func is not None:
            z = self._active_func(z)

        if self._flatten:
            z = z.flatten(2)

        self._output = z

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

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
