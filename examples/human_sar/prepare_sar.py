import os
import sys
import argparse
import numpy as np

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

import conf_cuhk_sar as conf

# Program arguments parser
data_txt = """
The input data. 'X' stands for image, 'A' stands for attribute, and 'S' stands
for segmentation.
"""

output_txt = """
If not specified, the output data will be saved as data_sar.pkl.
Otherwise it will be saved as data_sar_name.pkl.
"""

parser = argparse.ArgumentParser(description='Prepare the data')
parser.add_argument('-d', '--dataset', nargs='+', required=True,
                    choices=['Mix_SAR', 'CUHK_SAR'])
parser.add_argument('-i', '--input', nargs='+', required=True,
                    choices=['X', 'A', 'S'], help=data_txt)
parser.add_argument('-t', '--target', nargs='+', required=True,
                    choices=['X', 'A', 'S'], help=data_txt)
parser.add_argument('-o', '--output', nargs='?', default=None,
                    metavar='name', help=output_txt)

args = parser.parse_args()


def load_rawdata(dnames):
    from scipy.io import loadmat

    rawdata = []
    for dname in dnames:
        fpath = os.path.join(homepath, 'data', 'human_sar', dname + '.mat')
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
        if img.shape != (160, 80):
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
        # S[i] = (seg >= 0.5).astype(np.float32)
        # S[i] = imgproc.resize(S[i], [37, 17]).astype(np.float32)

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


def save_dataset(dataset, fpath):
    import cPickle

    with open(fpath, 'wb') as f:
        cPickle.dump(dataset, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    rawdata = load_rawdata(args.dataset)
    dataset = create_dataset(rawdata)

    out_file = 'data_sar.pkl' if args.output is None else \
               'data_sar_{0}.pkl'.format(args.output)
    save_dataset(dataset, out_file)
