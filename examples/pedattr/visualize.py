import os
import sys
import cPickle

homepath = os.path.join('..', '..')

if not homepath in sys.path:
    sys.path.insert(0, homepath)

from dlearn.visualization.vis_output import OutputVisualizer


if len(sys.argv) != 2:
    sys.exit('{0} cnn'.format(sys.argv[0]))
mdname = sys.argv[1]


def load_data():
    with open('data.pkl', 'rb') as f:
        dataset = cPickle.load(f)
    return dataset


def load_model():
    with open('model_{0}.pkl'.format(mdname), 'rb') as f:
        model = cPickle.load(f)
    return model

if __name__ == '__main__':
    dataset = load_data()
    model = load_model()
    visualizer = OutputVisualizer(model.input,
                                  model.blocks[0].output,
                                  input_shape=(3, 128, 48),
                                  output_shape=model.blocks[0].output_shape)

    x = dataset.train_x.get_value(borrow=True)
    visualizer.visualize(x[0])
