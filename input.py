import os

import sys
import tarfile
import pickle
import numpy as np

from six.moves import urllib
from tensorflow.contrib.learn.python.learn.datasets import base

DATA_DIR = 'data'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch >= self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def maybe_download_and_extract():
    dest_dir = DATA_DIR
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_dir)


def get_data(num_training=45000, num_validation=5000, num_test=1000):
    X_train, X_test, y_train, y_test = _read_data()

    # Shuffle the dataset for splitting out validation set
    # perm = np.arange(num_training + num_validation)
    # np.random.shuffle(perm)
    # X_train = X_train[perm]
    # y_train = y_train[perm]

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    train = DataSet(X_train, y_train)
    validation = DataSet(X_val, y_val)
    test = DataSet(X_test, y_test)

    return base.Datasets(train=train, validation=validation, test=test)


def _read_data():
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(DATA_DIR, 'cifar-10-batches-py', 'data_batch_%d' % (b,))
        X, y = _read_cifar10_file(f)
        xs.append(X)
        ys.append(y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    X_test, y_test = _read_cifar10_file(os.path.join(DATA_DIR, 'cifar-10-batches-py', 'test_batch'))
    return X_train, X_test, y_train, y_test


def _read_cifar10_file(filepath):
    print("Reading", filepath)
    with open(filepath, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        y = np.array(datadict[b'labels'])
        return X, y
