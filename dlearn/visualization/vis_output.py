import numpy as np
import theano

import gui


class OutputVisualizer(object):

    def __init__(self, input, output, input_shape, output_shape):
        super(OutputVisualizer, self).__init__()

        self._input_shape = (1,) + input_shape
        self._output_shape = output_shape

        self._input = theano.shared(
            np.zeros(self._input_shape, dtype=theano.config.floatX),
            borrow=True)

        self._f = theano.function(
            inputs=[], outputs=output,
            givens={
                input: self._input
            })

    def visualize(self, input, n_cols=8):
        self._input.set_value(input[np.newaxis, :], borrow=True)
        output = self._f().reshape(self._output_shape)
        gui.show_channels(output, n_cols)
