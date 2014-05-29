import os
import sys
import argparse
import numpy as np
import theano

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

from dlearn.visualization import show_channels
from dlearn.utils.serialize import load_data


# Program arguments parser
desctxt = """
Visualize segmentation result of testing data using learned model.
"""

dataset_txt = """
The dataset data_name.pkl.
"""

seg_txt = """
The segmentation model model_name.pkl.
"""

output_txt = """
The output folder name.
"""

parser = argparse.ArgumentParser(description=desctxt)
parser.add_argument('-d', '--dataset', nargs=1, required=True,
                    metavar='name', help=dataset_txt)
parser.add_argument('-s', '--segmentation', nargs=1, required=True,
                    metavar='name', help=seg_txt)
parser.add_argument('-o', '--output', nargs=1, required=True,
                    metavar='name', help=output_txt)
parser.add_argument('--ground-truth', action='store_true')

args = parser.parse_args()


def visualize(model, subset, folder, gt):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    f = theano.function(
        inputs=[model.input],
        outputs=model.blocks[3].output
    )

    X = subset.input

    output_shape = (1, 37, 17)

    y = f(X.cpu_data[0:100])

    if not gt:
        for i in xrange(100):
            print 'Saving figure {0}'.format(i)
            v = y[i].reshape(output_shape)
            show_channels(v, n_cols=1,
                          ofpath=os.path.join(folder, '{:04d}.png'.format(i)))
    else:
        S = subset.target
        for i in xrange(100):
            print 'Saving figure {0}'.format(i)
            v = np.vstack((y[i].reshape(output_shape), S.cpu_data[i:i + 1]))
            show_channels(v, n_cols=2,
                          ofpath=os.path.join(folder, '{:04d}.png'.format(i)))


if __name__ == '__main__':
    dataset_file = 'data_{0}.pkl'.format(args.dataset[0])
    seg_file = 'model_{0}.pkl'.format(args.segmentation[0])

    dataset = load_data(dataset_file)
    model = load_data(seg_file)
    visualize(model, dataset.test, args.output, args.ground_truth)
