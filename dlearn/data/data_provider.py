from glob import glob
from os.path import join as pj

from ..utils.math import create_shared
from ..utils.serialize import load_data

class Subset(object):
    def __init__(self, homepath, irange, pfunc=None):
        super(Subset, self).__init__()

        self._batch_files = [pj(homepath, 'data_batch_{}'.format(i))
                             for i in xrange(irange[0], irange[1] + 1)]
        self._pfunc = pfunc

        self._input = None
        self._target = None
        self.prepare(0)
        
    def prepare(self, index):
        input, target = load_data(self._batch_files[index])
        if self._pfunc is not None:
            input, target = self._pfunc(input, target)

        if self._input is None:
            self._input = [create_shared(X) for X in input]
            self._target = [create_shared(X) for X in target]
        else:
            for i, X in enumerate(input):
                self._input[i].set_value(X, borrow=True)
            for i, X in enumerate(target):
                self._target[i].set_value(X, borrow=True)

    @property
    def n_batches(self):
        return len(self._batch_files)

    @property
    def input(self):
        return self._input

    @property
    def target(self):
        return self._target

class DataProvider(object):
    def __init__(self, homepath, train_range, valid_range, test_range,
                 pfunc=None):
        super(DataProvider, self).__init__()

        self._train = Subset(homepath, train_range, pfunc)
        self._valid = Subset(homepath, valid_range, pfunc)
        self._test = Subset(homepath, test_range, pfunc)

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test

