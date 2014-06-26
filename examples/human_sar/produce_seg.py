import os
import sys
import argparse
import numpy as np
import theano
from scipy.io import loadmat, savemat

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

from dlearn.utils.serialize import load_data
from dlearn.utils import imgproc


# Program arguments parser
desctxt = """
Produce segmentation result of attribute dataset using learned model.
"""

seg_txt = """
The segmentation model model_name.pkl.
"""

output_txt = """
The output will be saved as name.mat.
"""

parser = argparse.ArgumentParser(description=desctxt)
parser.add_argument('-d', '--dataset', nargs=1, required=True,
                    choices=['Mix'])
parser.add_argument('-s', '--segmentation', nargs=1, required=True,
                    metavar='name', help=seg_txt)
parser.add_argument('-o', '--output', nargs=1, required=True,
                    metavar='name', help=output_txt)

args = parser.parse_args()


def load_rawdata(dnames):
    rawdata = []
    for dname in dnames:
        fpath = os.path.join(homepath, 'data', 'human_attribute', dname)
        matdata = loadmat(fpath)
        m, n = matdata['images'].shape
        rawdata.extend([matdata['images'][i, 0] for i in xrange(m)])

    return rawdata


def proc_rawdata(rawdata):
    def imgresize(img):
        img = imgproc.resize(img, [160, 80], keep_ratio='height')
        return img

    def imgprep(img):
        img = imgproc.subtract_luminance(img)
        img = np.rollaxis(img, 2)
        return (img / 100.0).astype(np.float32)

    I = [imgresize(img) for img in rawdata]
    X = [imgprep(img) for img in I]
    X = np.asarray(X)
    X = X - X.mean(axis=0)
    return I, X


def produce_segmentation(data, model):
    f = theano.function(
        inputs=[model.input],
        outputs=model.blocks[3].output
    )

    m = data.shape[0]
    output = [0] * m

    for i in xrange(m):
        S = f(data[i:i + 1]).reshape([37, 17])
        output[i] = imgproc.resize(S, [160, 80])

    return np.asarray(output)


def save_segmentation(images, output, fpath):
    savemat(fpath, {'images': images, 'segmentations': output})


if __name__ == '__main__':
    seg_file = 'model_{0}.pkl'.format(args.segmentation[0])
    out_file = '{0}.mat'.format(args.output[0])

    data = load_rawdata(args.dataset)
    images, data = proc_rawdata(data)
    model = load_data(seg_file)

    output = produce_segmentation(data, model)

    save_segmentation(images, output, out_file)
