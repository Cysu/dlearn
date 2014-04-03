class Block(object):

    """Construct a base class to represent an input-output black box.

    Attributes
    ----------
    parameters : list of theano.tensor
        The block parameters.
    input : theano.tensor or list of theano.tensor
        The input symbol(s) of the block.
    output : theano.tensor or list of theano.tensor
        The output symbol(s) of the block.

    Parameters
    ----------
    input : theano.tensor or list of theano.tensor
        The input symbol(s) of the block.

    """

    def __init__(self, input):
        self._params = []
        self._input = input
        self._output = input

    @property
    def parameters(self):
        return self._params

    @parameters.setter
    def parameters(self, value):
        self._params = value

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value
