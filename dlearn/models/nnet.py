from .block import Block


class NeuralNet(Block):

    r"""Construct a neural network with several blocks.

    The neural network is just a wrapper of blocks and some training variables.
    It helps to serialize them more conveniently. It is not in charge of
    connecting them. The connection can be non-sequential.

    Attributes
    ----------
    blocks : list of Block
        The blocks of the neural network.
    target : theano.tensor or list of theano.tensor
        The target symbol(s) of the `cost` and `error`.
    cost : theano.tesnor.scalar
        The cost to be used for training.
    error : theano.tensor.scalar
        The error to be used for testing.
    inc_updates : list of update tensors
        The incremental updates. :math:`inc\gets momentum\times inc - lr\times
        gradient`.
    param_updates : list of update tensors
        Then parameter updates. :math:`param\gets param+inc`.

    Parameters
    ----------
    blocks : list of Block
        The blocks of the neural network.
    input : theano.tensor or list of theano.tensor
        The input symbol(s) of the neural network.
    output : theano.tensor or list of theano.tensor
        The output symbol(s) of the neural network.

    """

    def __init__(self, blocks, input, output):
        super(NeuralNet, self).__init__(input, None)

        self._blocks = blocks

        self._params = list(reduce(lambda x, y: x | y,
                                   [set(b.parameters) for b in self._blocks]))

        self._output = output

        self._cost = None
        self._error = None
        self._target = None
        self._consts = []
        self._inc_updates = []
        self._param_updates = []

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value

    @property
    def blocks(self):
        return self._blocks

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self._cost = value

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, value):
        self._error = value

    @property
    def consts(self):
        return self._consts

    @consts.setter
    def consts(self, value):
        self._consts = value

    @property
    def inc_updates(self):
        return self._inc_updates

    @inc_updates.setter
    def inc_updates(self, value):
        self._inc_updates = value

    @property
    def param_updates(self):
        return self._param_updates

    @param_updates.setter
    def param_updates(self, value):
        self._param_updates = value

    def get_norm(self, l):
        """Return the summation of all the block norms.

        Parameters
        ----------
        l : int
            The L?-norm.

        Returns
        -------
        out : theano.tensor.scalar
            The summation of all the block norms.

        """
        ret = []
        for b in self._blocks:
            if 'get_norm' in dir(b) and callable(getattr(b, 'get_norm')):
                ret.append(b.get_norm(l))
        return sum(ret)

    def get_squared_L2(self):
        ret = []
        for b in self._blocks:
            if 'get_squared_L2' in dir(b) and \
                    callable(getattr(b, 'get_squared_L2')):
                ret.append(b.get_squared_L2())
        return sum(ret)

