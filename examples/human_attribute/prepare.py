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
            rawdata.append([
                matdata['images'][i, 0],
                matdata['attributes'][i, 0].ravel()
            ])

        fpath = os.path.join(homepath, 'data', 'human_segmentation', dname)
        matdata = loadmat(fpath)
        for i in xrange(m):
            rawdata[i].append(matdata['segmentations'][i, 0])

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
        img = imgproc.resize(img, [17, 7])
        return img.astype(np.float32)

    m = len(rawdata)
    X = [0] * (2 * m)
    Y = [0] * (2 * m)
    S = [0] * (2 * m)

    for i, (img, attr, seg) in enumerate(rawdata):
        X[i * 2] = imgprep(img)
        X[i * 2 + 1] = X[i * 2][:, :, ::-1].copy()
        Y[i * 2] = choose_multival(attr, 'Upper Body Colors')
        Y[i * 2 + 1] = Y[i * 2]
        S[i * 2] = choose_segment(seg, 'Upper')
        S[i * 2 + 1] = S[i * 2][:, ::-1].copy()
        S[i * 2] = S[i * 2].ravel()
        S[i * 2 + 1] = S[i * 2 + 1].ravel()

    X = np.asarray(X)
    Y = np.asarray(Y)
    S = np.asarray(S)

    X = X - X.mean(axis=0)

    dataset = Dataset([X, S], Y)
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
