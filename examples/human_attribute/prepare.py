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
        fpath_a = os.path.join(homepath, 'data', 'human_attribute', dname)
        matdata_a = loadmat(fpath_a)

        fpath_s = os.path.join(homepath, 'data', 'human_segmentation', dname)
        matdata_s = loadmat(fpath_s)

        m, n = matdata_a['images'].shape
        for i in xrange(m):
            rawdata.append([
                matdata_a['images'][i, 0],
                matdata_a['attributes'][i, 0].ravel(),
                matdata_s['segmentations'][i, 0]
            ])

    return rawdata


def create_dataset(rawdata):
    from dlearn.data.dataset import Dataset
    from dlearn.utils import imgproc

    def imgprep(img):
        img = imgproc.subtract_luminance(img)
        img = np.rollaxis(img, 2)
        return (img / 100.0).astype(np.float32)

    def choose_unival(attr, title):
        ind = conf.unival_titles.index(title)
        vals = conf.unival[ind]
        ind = [conf.names.index(v) for v in vals]
        return np.where(attr[ind] == 1)[0][0]

    def choose_multival(attr, title):
        ind = conf.multival_titles.index(title)
        vals = conf.multival[ind]
        ind = [conf.names.index(v) for v in vals]
        return attr[ind].astype(np.float32)

    def choose_segment(seg, title):
        val = conf.segment_vals[title]
        img = (seg == val).astype(np.float32)
        return img.astype(np.float32)

    m = len(rawdata)
    X = [0] * (2 * m)
    A = [0] * (2 * m)
    S = [0] * (2 * m)

    i = 0
    for (img, attr, seg) in rawdata:
        X[i] = imgprep(img)
        A[i] = choose_multival(attr, 'Upper Body Colors')
        S[i] = choose_segment(seg, 'Upper')

        # Mirror
        X[i + 1] = X[i][:, :, ::-1].copy()
        A[i + 1] = A[i].copy()
        S[i + 1] = S[i][:, ::-1].copy()

        i += 2

    X = np.asarray(X)
    A = np.asarray(A)
    S = np.asarray(S)

    X = X - X.mean(axis=0)

    dataset = Dataset([X, S], A)
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
