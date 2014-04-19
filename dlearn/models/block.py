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
    dropout_input : None, theano.tensor, or list of theano.tensor
        None if no dropout input symbol is specified, and the `input` will be
        used as `dropout_input`. *Read-only*.
    dropout_output : None, theano.tensor, or list of theano.tensor
        None if no dropout for this layer, and the `output` will be used as
        `dropout_output`. *Read-only*.

    Parameters
    ----------
    input : theano.tensor or list of theano.tensor
        The input symbol(s) of the block.
    dropout_input : None, theano.tensor, or list of theano.tensor, optional
        None if no dropout input symbol is specified, and the `input` will be
        used as `dropout_input`. Default is None.

    """

    def __init__(self, input, dropout_input=None):
        super(Block, self).__init__()

        self._params = []
        self._input = input
        self._output = input

        if dropout_input is None:
            self._dropout_input = self._input
        else:
            self._dropout_input = dropout_input

        self._dropout_output = self._dropout_input

    @property
    def parameters(self):
        return self._params

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def dropout_input(self):
        return self._dropout_input

    @property
    def dropout_output(self):
        return self._dropout_output
