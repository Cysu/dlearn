import os
import sys
import numpy as np

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

import conf_cuhk_sar as conf


def load_rawdata():
    from scipy.io import loadmat

    rawdata = []
    for dname in conf.datasets:
        fpath = os.path.join(homepath, 'data', 'human_attribute', dname)
        matdata = loadmat(fpath)

        m, n = matdata['images'].shape
        for i in xrange(m):
            rawdata.append((
                matdata['images'][i, 0],
                matdata['attributes'][i, 0].ravel()
            ))

    return rawdata


def create_dataset(rawdata):
    from dlearn.data.dataset import Dataset
    from dlearn.utils import imgproc

    def imgprep(img):
        img = imgproc.subtract_luminance(img)
        img = np.rollaxis(img, 2)
        return img / 100.0

    def choose_unival(attr, title):
        ind = conf.unival_titles.index(title)
        vals = conf.unival[ind]
        ind = [conf.names.index(v) for v in vals]
        return np.where(attr[ind] == 1)[0][0]

    def choose_multival(attr, title):
        ind = conf.multival_titles.index(title)
        vals = conf.multival[ind]
        ind = [conf.names.index(v) for v in vals]
        return attr[ind]

    m = len(rawdata)
    X = [0] * (2 * m)
    Y = [0] * (2 * m)

    for i, (img, attr) in enumerate(rawdata):
        X[i * 2] = imgprep(img)
        X[i * 2 + 1] = X[i * 2][:, :, ::-1].copy()
        Y[i * 2] = choose_multival(attr, 'Upper Body Colors')
        Y[i * 2 + 1] = Y[i * 2]

    X = np.asarray(X)
    Y = np.asarray(Y)

    X = X - X.mean(axis=0)

    dataset = Dataset(X, Y)
    dataset.split(0.7, 0.2)

    return dataset


def save_dataset(dataset):
    import cPickle

    with open('data.pkl', 'wb') as f:
        cPickle.dump(dataset, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    rawdata = load_rawdata()
    dataset = create_dataset(rawdata)
    save_dataset(dataset)
