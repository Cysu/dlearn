from ..utils.math import nprng, create_shared


class Dataset(object):

    """Construct a dataset to provide training, validation, and testing data.

    The dataset is just a wrapper of these data.

    Attributes
    ----------
    X : numpy.ndarray
    Y : numpy.ndarray
    train_x : theano.tensor(shared)
    train_y : theano.tensor(shared)
    valid_x : theano.tensor(shared)
    valid_y : theano.tensor(shared)
    test_x : theano.tensor(shared)
    test_y : theano.tensor(shared)
    
    """

    def __init__(self, X=None, Y=None):
        super(Dataset, self).__init__()

        self.X = X
        self.Y = Y

    def split(self, train_ratio, valid_ratio,
              datatype_x='float32', datatype_y='float32'):
        n = self.X.shape[0]

        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        p = nprng.permutation(n)

        self.train_ind = p[0: n_train]
        self.valid_ind = p[n_train: n_train + n_valid]
        self.test_ind = p[n_train + n_valid:]

        self.train_x = create_shared(self.X[self.train_ind], datatype_x)
        self.train_y = create_shared(self.Y[self.train_ind], datatype_y)
        self.valid_x = create_shared(self.X[self.valid_ind], datatype_x)
        self.valid_y = create_shared(self.Y[self.valid_ind], datatype_y)
        self.test_x = create_shared(self.X[self.test_ind], datatype_x)
        self.test_y = create_shared(self.Y[self.test_ind], datatype_y)
