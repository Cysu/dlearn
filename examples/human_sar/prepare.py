import os
import sys
import argparse
import numpy as np

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

import conf_cuhk_sar as conf


# Program arguments parser
parser = argparse.ArgumentParser(description='Prepare the data')
helptxt = """
The input data. 'X' stands for image, 'A' stands for attribute, and 'S' stands
for segmentation.
"""
parser.add_argument('-i', '--input', nargs='+', required=True,
                    choices=['X', 'A', 'S'], help=helptxt)
parser.add_argument('-t', '--target', nargs='+', required=True,
                    choices=['X', 'A', 'S'], help=helptxt)

args = parser.parse_args()


def load_rawdata():
    from scipy.io import loadmat

    rawdata = []
    for dname in ['Mix_SAR']:
        fpath = os.path.join(homepath, 'data', 'human_sar', dname)
        matdata = loadmat(fpath)
        m, n = matdata['images'].shape
        for i in xrange(m):
            rawdata.append([
                matdata['images'][i, 0],
                matdata['attributes'][i, 0].ravel(),
                matdata['segmentations'][i, 0]
            ])

    return rawdata


def create_dataset(rawdata):
    from dlearn.data.dataset import Dataset
    from dlearn.utils import imgproc

    def imgprep(img):
        img = imgproc.resize(img, [160, 80], keep_ratio='height')
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

    def choose_seg(seg, title):
        val = conf.seg_pix[title]
        img = (seg == val).astype(np.float32)
        img = imgproc.resize(img, [37, 17])
        return img.astype(np.float32)

    m = len(rawdata)
    X = [0] * (2 * m)
    A = [0] * (2 * m)
    S = [0] * (2 * m)

    i = 0
    for (img, attr, seg) in rawdata:
        X[i] = imgprep(img)
        A[i] = choose_attr_mul(attr, 'Upper Body Colors')
        S[i] = choose_seg(seg, 'Upper')

        # Mirror
        X[i + 1] = X[i][:, :, ::-1].copy()
        A[i + 1] = A[i].copy()
        S[i + 1] = S[i][:, ::-1].copy()

        i += 2

    X = np.asarray(X)
    A = np.asarray(A)
    S = np.asarray(S)

    X = X - X.mean(axis=0)

    def parse_list(arglist):
        d = {'X': X, 'A': A, 'S': S}
        l = map(lambda x: d[x], arglist)
        return l[0] if len(l) == 1 else l

    dataset = Dataset(parse_list(args.input), parse_list(args.target))
    dataset.split(0.7, 0.2)

    return dataset


def save_dataset(dataset):
    import cPickle

    with open('data_mix.pkl', 'wb') as f:
        cPickle.dump(dataset, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    rawdata = load_rawdata()
    dataset = create_dataset(rawdata)
    save_dataset(dataset)
