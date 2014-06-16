import numpy as np

from .hidata import HierarData
from ..utils.math import nprng


def _get_n(data):
    return data[0].shape[0] if isinstance(data, list) else data.shape[0]


def _combine(*args):
    if isinstance(args[0], list):
        data = []
        for j in xrange(len(args[0])):
            data.append(np.concatenate([d[j] for d in args], axis=0))
    else:
        data = np.concatenate([d for d in args], axis=0)

    return data


class Subset(object):

    def __init__(self, input=None, target=None):
        self.input = input
        self.target = target

    def prepare(self, irange):
        if isinstance(self.input, list):
            for x in self.input:
                x.prepare(irange)
        else:
            self.input.prepare(irange)

        if isinstance(self.target, list):
            for x in self.target:
                x.prepare(irange)
        else:
            self.target.prepare(irange)


class Dataset(object):

    """Construct a dataset to provide training, validation, and testing data.

    The dataset is just a wrapper of these data. All the data are stored in CPU
    memory.

    Attributes
    ----------
    input : numpy.ndarray or list of numpy.ndarray
        The whole input data. If there is only one category of input data, then
        alongside the first dimension of the numpy.ndarray are input samples.
        Otherwise, multiple categories of input data form a list of
        numpy.ndarray. *Read-only*.

    target : numpy.ndarray or list of numpy.ndarray
        The whole target data. If there is only one category of target data,
        then alongside the first dimension of the numpy.ndarray are target
        samples. Otherwise, multiple categories of target data form a list of
        numpy.ndarray. *Read-only*.

    train, valid, test : Wrapper
        These are Wrapper objects for training, validation, and testing data.
        Each contains three fields: input, target, and index. Field input is a
        subset of the whole input data; field target is a subset of the whole
        target data; field index is a numpy vector storing the index of subset
        samples in the whole data. *Read-only*.

    Parameters
    ----------
    input : numpy.ndarray or list of numpy.ndarray
        The whole input data. If there is only one category of input data, then
        alongside the first dimension of the numpy.ndarray are input samples.
        Otherwise, multiple categories of input data form a list of
        numpy.ndarray.

    target : numpy.ndarray or list of numpy.ndarray
        The whole target data. If there is only one category of target data,
        then alongside the first dimension of the numpy.ndarray are target
        samples. Otherwise, multiple categories of target data form a list of
        numpy.ndarray.
    """

    def __init__(self, input=None, target=None,
                 train=None, valid=None, test=None,
                 limit=None):
        super(Dataset, self).__init__()

        self._limit = limit
        self._train = Subset()
        self._valid = Subset()
        self._test = Subset()

        if input is not None and target is not None:
            self._input = input
            self._target = target
        elif train is not None and valid is not None and test is not None:
            self._input = _combine(train.input, valid.input, test.input)
            self._target = _combine(train.target, valid.target, test.target)

            if isinstance(train.input, list):
                self._train.input = \
                    [HierarData(X, limit=self._limit) for X in train.input]
                self._valid.input = \
                    [HierarData(X, limit=self._limit) for X in valid.input]
                self._test.input = \
                    [HierarData(X, limit=self._limit) for X in test.input]
            else:
                self._train.input = HierarData(train.input, limit=self._limit)
                self._valid.input = HierarData(valid.input, limit=self._limit)
                self._test.input = HierarData(test.input, limit=self._limit)

            if isinstance(train.target, list):
                self._train.target = \
                    [HierarData(X, limit=self._limit) for X in train.target]
                self._valid.target = \
                    [HierarData(X, limit=self._limit) for X in valid.target]
                self._test.target = \
                    [HierarData(X, limit=self._limit) for X in test.target]
            else:
                self._train.target = HierarData(
                    train.target, limit=self._limit)
                self._valid.target = HierarData(
                    valid.target, limit=self._limit)
                self._test.target = HierarData(test.target, limit=self._limit)

            n_train = _get_n(train.input)
            n_valid = _get_n(valid.input)
            n_test = _get_n(test.input)

            self._train_ind = np.arange(n_train)
            self._valid_ind = np.arange(n_train, n_train + n_valid)
            self._test_ind = np.arange(n_train + n_valid,
                                       n_train + n_valid + n_test)
        else:
            raise ValueError("Invalid combination of arguments")

    def split(self, train_ratio, valid_ratio):
        n = _get_n(self._input)

        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        p = nprng.permutation(n)

        self._train_ind = p[0: n_train]
        self._valid_ind = p[n_train: n_train + n_valid]
        self._test_ind = p[n_train + n_valid:]

        if isinstance(self._input, list):
            self._train.input = \
                [HierarData(X[self._train_ind], limit=self._limit)
                 for X in self._input]
            self._valid.input = \
                [HierarData(X[self._valid_ind], limit=self._limit)
                 for X in self._input]
            self._test.input = \
                [HierarData(X[self._test_ind], limit=self._limit)
                 for X in self._input]
        else:
            self._train.input = HierarData(self._input[self._train_ind],
                                           limit=self._limit)
            self._valid.input = HierarData(self._input[self._valid_ind],
                                           limit=self._limit)
            self._test.input = HierarData(self._input[self._test_ind],
                                          limit=self._limit)

        if isinstance(self._target, list):
            self._train.target = \
                [HierarData(Y[self._train_ind], limit=self._limit)
                 for Y in self._target]
            self._valid.target = \
                [HierarData(Y[self._valid_ind], limit=self._limit)
                 for Y in self._target]
            self._test.target = \
                [HierarData(Y[self._test_ind], limit=self._limit)
                 for Y in self._target]
        else:
            self._train.target = HierarData(self._target[self._train_ind],
                                            limit=self._limit)
            self._valid.target = HierarData(self._target[self._valid_ind],
                                            limit=self._limit)
            self._test.target = HierarData(self._target[self._test_ind],
                                           limit=self._limit)

    @property
    def input(self):
        return self._input

    @property
    def target(self):
        return self._target

    @property
    def train_ind(self):
        return self._train_ind

    @property
    def valid_ind(self):
        return self._valid_ind

    @property
    def test_ind(self):
        return self._test_ind

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test
