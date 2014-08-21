import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d

try:
    from theano.sandbox.cuda.basic_ops import gpu_contiguous
    from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
except ImportError:
    use_convnet = False
    print 'Use theano convolution'
else:
    use_convnet = True
    print 'Try cuda-convnet'

from .block import Block
from ..utils import actfuncs
from ..utils.math import nprng, dropout


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
    dropout_input : None, theano.tensor, or list of theano.tensor, optional
        None if no dropout input symbol is specified, and the `input` will be
        used as `dropout_input`. Default is None.
    dropout_ratio : None or float, optional
        None if no dropout for this layer, otherwise the `dropout_ratio` portion
        of output will be dropout. Default is None.
    active_func : None, theano.Op or function, optional
        If None, then no active function will be applied to the output,
        otherwise the specified active function will be applied. Default is
        None.
    W : None or theano.matrix(shared), optional
        If None, the weight matrix will be intialized randomly, otherwise it
        will be set to the specified value. Default is None.
    b : None or theano.vector(shared), optional
        If None, the bias vector will be initialized as zero, otherwise it will
        be set to the specified value. Default is None.

    """

    def __init__(self, input, input_shape, output_shape,
                 dropout_input=None, dropout_ratio=None,
                 active_func=None, W=None, b=None, const_params=False):
        super(FullConnLayer, self).__init__(input, dropout_input)

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
        self._dropout_ratio = dropout_ratio

        # Initialize parameters
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

        if not const_params:
            self._params = [self._W, self._b]

        # Compute output and dropout output
        def f(x):
            z = T.dot(x, self._W) + self._b
            return z if self._active_func is None else self._active_func(z)

        self._output = f(self._input)
        self._dropout_output = f(self._dropout_input)

        if self._dropout_ratio is not None and self._dropout_ratio > 0:
            self._output = (1.0 - self._dropout_ratio) * self._output
            self._dropout_output = dropout(self._dropout_output,
                                           self._dropout_ratio)

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

    def get_squared_L2(self):
        return T.sum(self._W ** 2)


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
    border_mode : str, optional
        Suppose the image size is :math:`m` and the filter size is :math:`n`. If
        ``'valid'``, then the response size is :math:`m-n+1`. If ``'full'``,
        then the response size is :math:`m+n-1`. Other modes are not supported.
        Default is ``'valid'``.
    pool_shape : None or list/tuple of int, optional
        If None, then no max-pooling will be performed, otherwise the shape of
        the max-pooling region should be (n_rows, n_cols). Default if None.
    dropout_input : None, theano.tensor, or list of theano.tensor, optional
        None if no dropout input symbol is specified, and the `input` will be
        used as `dropout_input`. Default is None.
    dropout_ratio : None or float, optional
        None if no dropout for this layer, otherwise the `dropout_ratio` portion
        of output will be dropout. Default is None.
    active_func : None, theano.Op or function, optional
        If None, then no active function will be applied to the output,
        otherwise the specified will be applied. Default is None.
    flatten : bool, optional
        If False, then output will not be flattened, otherwise the output of
        each sample will be flattened into a vector. Default is False.
    W : None or theano.matrix(shared), optional
        If None, the convolutional matrix will be intialized randomly, otherwise
        it will be set to the specified value. Default is None.
    b : None, float, or theano.vector(shared), optional
        If None, the bias vector will be initialized as zero. If float, the bias
        vector will be constant. Otherwise it will be set to the specified
        value. Default is None.

    """

    def __init__(self, input, input_shape, filter_shape, border_mode='valid',
                 pool_shape=None,
                 dropout_input=None, dropout_ratio=None,
                 active_func=None, flatten=False,
                 W=None, b=None, const_params=False):
        super(ConvPoolLayer, self).__init__(input, dropout_input)

        if border_mode not in ['valid', 'full', 'same']:
            raise ValueError("border_mode value error")

        self._input_shape = input_shape
        self._filter_shape = filter_shape
        self._border_mode = border_mode
        self._pool_shape = pool_shape if pool_shape is not None else (1, 1)
        self._dropout_ratio = dropout_ratio
        self._active_func = active_func
        self._flatten = flatten

        # Compute output shape
        if self._border_mode == 'valid':
            n_rows = (self._input_shape[-2] - self._filter_shape[-2] + 1)
            n_cols = (self._input_shape[-1] - self._filter_shape[-1] + 1)
        elif self._border_mode == 'full':
            n_rows = (self._input_shape[-2] + self._filter_shape[-2] - 1)
            n_cols = (self._input_shape[-1] + self._filter_shape[-1] - 1)
        else:
            n_rows = self._input_shape[-2]
            n_cols = self._input_shape[-1]

        self._output_shape = (
            self._filter_shape[0],
            n_rows // self._pool_shape[0],
            n_cols // self._pool_shape[1]
        )
        if flatten:
            self._output_shape = np.prod(self._output_shape)

        satisfy_convnet = use_convnet and \
            self._filter_shape[0] % 16 == 0 and \
            self._filter_shape[2] == self._filter_shape[3] and \
            (self._input_shape[0] <= 4 or self._input_shape[0] % 4 == 0)

        if use_convnet and not satisfy_convnet:
            print "Break the convnet constraints. Fallback to theano conv."

        if not satisfy_convnet and self._border_mode == 'same':
            raise ValueError("Cannot use 'same' border mode when using theano conv")

        # Initialize parameters
        if W is None:
            fan_in = np.prod(self._filter_shape[1:])
            fan_out = self._filter_shape[0] * \
                np.prod(self._filter_shape[2:]) / np.prod(self._pool_shape)

            W_bound = np.sqrt(6.0 / (fan_in + fan_out))

            if active_func == actfuncs.sigmoid:
                W_bound *= 4

            init_W = np.asarray(nprng.uniform(low=-W_bound, high=W_bound,
                                              size=self._filter_shape),
                                dtype=theano.config.floatX)

            self._W = theano.shared(value=init_W, borrow=True)
        else:
            self._W = W

        if b is None:
            init_b = np.zeros((filter_shape[0],), dtype=theano.config.floatX)

            self._b = theano.shared(value=init_b, borrow=True)
        else:
            self._b = b

        if not const_params:
            if isinstance(self._b, float):
                self._params = [self._W]
            else:
                self._params = [self._W, self._b]

        # Compute output and dropout output
        def f(x):
            if not satisfy_convnet:
                z = T.nnet.conv.conv2d(input=x, filters=self._W,
                                       filter_shape=self._filter_shape,
                                       border_mode=self._border_mode)
            else:
                if self._border_mode == 'valid':
                    pad = 0
                elif self._border_mode == 'same':
                    pad = (self._filter_shape[2] - 1) // 2
                else:
                    pad = self._filter_shape[2] - 1

                conv_op = FilterActs(stride=1, partial_sum=1, pad=pad)
                x = gpu_contiguous(x.dimshuffle(1, 2, 3, 0))
                W = gpu_contiguous(self._W.dimshuffle(1, 2, 3, 0))
                z = conv_op(x, W).dimshuffle(3, 0, 1, 2)

            if self._pool_shape != (1, 1):
                z = max_pool_2d(
                    input=z, ds=self._pool_shape,
                    ignore_border=True)

            if isinstance(self._b, float):
                z = z + self._b
            else:
                z = z + self._b.dimshuffle('x', 0, 'x', 'x')

            if self._active_func is not None:
                z = self._active_func(z)

            if self._flatten:
                z = z.flatten(2)

            return z

        self._output = f(self._input)
        self._dropout_output = f(self._dropout_input)

        if self._dropout_ratio is not None and self._dropout_ratio > 0:
            self._output = (1.0 - self._dropout_ratio) * self._output
            self._dropout_output = dropout(self._dropout_output,
                                           self._dropout_ratio)

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

    def get_squared_L2(self):
        return T.sum(self._W ** 2)


class RegMultitaskLayer(Block):
    def __init__(self, input, input_shape, output_shape,
                 dropout_input=None, dropout_ratio=None, active_func=None,
                 w0=None, b0=None, W=None, b=None, const_params=False):
        super(RegMultitaskLayer, self).__init__(input, dropout_input)

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
        self._dropout_ratio = dropout_ratio

        # Initialize parameters
        if w0 is None:
            w0_bound = np.sqrt(6.0 / (self._input_shape + self._output_shape))

            if active_func == actfuncs.sigmoid:
                w0_bound *= 4

            init_w0 = np.asarray(
                nprng.uniform(low=-w0_bound, high=w0_bound,
                              size=(self._input_shape, 1)),
                dtype=theano.config.floatX)

            self._w0 = theano.shared(value=init_w0, borrow=True)
        else:
            self._w0 = w0

        if b0 is None:
            init_b0 = np.asarray(0, theano.config.floatX)

            self._b0 = theano.shared(value=init_b0, borrow=True)
        else:
            self._b0 = b0

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

        if not const_params:
            self._params = [self._w0, self._b0, self._W, self._b]

        # Compute output and dropout output
        def f(x):
            z0 = T.dot(x, self._w0) + self._b0
            z = T.dot(x, self._W) + self._b
            z = z + z0.repeat(self._output_shape, axis=1)
            return z if self._active_func is None else self._active_func(z)

        self._output = f(self._input)
        self._dropout_output = f(self._dropout_input)

        if self._dropout_ratio is not None and self._dropout_ratio > 0:
            self._output = (1.0 - self._dropout_ratio) * self._output
            self._dropout_output = dropout(self._dropout_output,
                                           self._dropout_ratio)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def get_squared_L2_W(self):
        return T.sum(self._W ** 2)

    def get_squared_L2_w0(self):
        return T.sum(self._w0 ** 2)

    def get_regularization(self):
        inner_product = self._w0.repeat(self._output_shape, axis=1) * self._W
        inner_product = inner_product.sum(axis=0)
        return T.sum(inner_product ** 2)
