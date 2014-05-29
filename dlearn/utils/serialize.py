import cPickle


def load_data(fpath):
    with open(fpath, 'rb') as f:
        data = cPickle.load(f)
    return data


def save_data(data, fpath):
    with open(fpath, 'wb') as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)
