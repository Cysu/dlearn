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
    cost : theano.tesnor.scalar
        The cost to be used for training. Default is None.
    error : theano.tensor.scalar
        The error to be used for testing. Default is None.
    inc_updates : list of update tensors
        The incremental updates. :math:`inc\gets momentum\times inc - lr\times
        gradient`. Default is empty list.
    param_updates : list of update tensors
        Then parameter updates. :math:`param\gets param+inc`. Default is empty
        list.

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
        super(NeuralNet, self).__init__(input)

        self._blocks = blocks

        self._params = reduce(lambda x, y: x | y,
                              [set(b.parameters) for b in self._blocks])

        self._output = output

        self._cost = None
        self._error = None
        self._inc_updates = []
        self._param_updates = []

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
        return sum([b.get_norm(l) for b in self._blocks])
