import os
import sys
import cPickle
import numpy as np
import theano
from scipy.io import loadmat, savemat

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)


def load_rawdata():
    rawdata = []
    for dname in ['Mix']:
        fpath = os.path.join(homepath, 'data', 'human_attribute', dname)
        matdata = loadmat(fpath)
        m, n = matdata['images'].shape
        rawdata.extend([matdata['images'][i, 0] for i in xrange(m)])

    return rawdata


def proc_rawdata(rawdata):
    from dlearn.utils import imgproc

    def imgprep(img):
        img = imgproc.resize(img, [160, 80], keep_ratio='height')
        img = imgproc.subtract_luminance(img)
        img = np.rollaxis(img, 2)
        return (img / 100.0).astype(np.float32)

    X = [imgprep(img) for img in rawdata]
    X = np.asarray(X)
    X = X - X.mean(axis=0)
    return X


def load_model():
    with open('model_segcnn.pkl', 'rb') as f:
        model = cPickle.load(f)
    return model


def produce_segmentation(data, model):
    f = theano.function(
        inputs=[model.input],
        outputs=model.output,
        on_unused_input='ignore'
    )

    m = data.shape[0]
    output = [0] * m

    for i in xrange(m):
        output[i] = f(data[i:i + 1]).reshape([37, 17])

    return np.asarray(output)


def save_segmentation(output, fpath):
    savemat(fpath, {'segmentation': output})


if __name__ == '__main__':
    data = load_rawdata()
    data = proc_rawdata(data)
    model = load_model()

    output = produce_segmentation(data, model)
    save_segmentation(output, 'Mix_segmentation.mat')
