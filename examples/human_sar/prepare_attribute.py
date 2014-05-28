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
    for dname in ['Mix']:
        fpath = os.path.join(homepath, 'data', 'human_attribute', dname)
        matdata = loadmat(fpath)
        m, n = matdata['images'].shape
        for i in xrange(m):
            rawdata.append([
                matdata['images'][i, 0],
                matdata['attributes'][i, 0].ravel()
            ])

    return rawdata


def create_dataset(rawdata):
    from dlearn.data.dataset import Dataset
    from dlearn.utils import imgproc

    def imgprep(img):
        img = imgproc.subtract_luminance(img)
        img = np.rollaxis(img, 2)
        return (img / 100.0).astype(np.float32)

    def choose_attr_uni(attr, title):
        ind = conf.attr_uni_titles.index(title)
        vals = conf.attr_uni[ind]
        ind = [conf.attr_names.index(v) for v in vals]
        return np.where(attr[ind] == 1)[0][0]

    def choose_attr_mul(attr, title):
        ind = conf.attr_mul_titles.index(title)
        vals = conf.attr_mul[ind]
        ind = [conf.attr_names.index(v) for v in vals]
        return attr[ind].astype(np.float32)

    m = len(rawdata)
    X = [0] * m
    A = [0] * m

    i = 0
    for (img, attr) in rawdata:
        X[i] = imgprep(img)
        A[i] = choose_attr_mul(attr, 'Upper Body Colors')
        i += 1

    X = np.asarray(X)
    A = np.asarray(A)

    X = X - X.mean(axis=0)

    dataset = Dataset(X, A)
    dataset.split(0.7, 0.2)

    return dataset


def save_dataset(dataset):
    import cPickle

    with open('data_attribute.pkl', 'wb') as f:
        cPickle.dump(dataset, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    rawdata = load_rawdata()
    dataset = create_dataset(rawdata)
    save_dataset(dataset)
