class Block(object):

    """Construct a base class to represent an input-output black box.

    Attributes
    ----------
    parameters : list of theano.tensor
        The block parameters. *Read-only*.
    input : theano.tensor or list of theano.tensor
        The input symbol(s) of the block. *Read-only*.
    output : theano.tensor or list of theano.tensor
        The output symbol(s) of the block. *Read-only*.

    Parameters
    ----------
    input : theano.tensor or list of theano.tensor
        The input symbol(s) of the block.

    """

    def __init__(self, input):
        super(Block, self).__init__()

        self._params = []
        self._input = input
        self._output = input

    @property
    def parameters(self):
        return self._params

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output
