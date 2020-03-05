from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import os
import warnings
import logging
import matplotlib.pyplot as plt
from absl import flags
from absl import app
from tqdm import tqdm
import pickle as cPickle
import urllib.request

flags.DEFINE_float('learning_rate',
                   default=0.001,
                   help='Initial learning rate.')
flags.DEFINE_integer('num_epochs',
                     default=10,
                     help='Number of training steps to run.')
flags.DEFINE_integer('batch_size',
                     default=128,
                     help='Batch size.')
flags.DEFINE_string('data_dir',
                    default="./datasets",
                    help='Directory where data is stored (if using real data).')
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getenv('TEST_TMPDIR', './model'),
                         'cifar10/'),
    help="Directory to put the model's fit.")
flags.DEFINE_integer('viz_steps',
                     default=400,
                     help='Frequency at which save visualizations.')

flags.DEFINE_integer('epochs',
                     default=5000,
                     help='number of maximum epochs')

flags.DEFINE_string('cifar10_link',
                    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                    'link download cifar10 dataset')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def my_hook(t):
    """
    Wraps tqdm instance
    :param t:
    :return:
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """

        :param b:       int option
        :param bsize:   int
        :param tsize:
        :return:
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner

def get_data():
    """

    :param input_data:
    :return:
    """
    if os.path.isdir(FLAGS.data_dir) == False:
        os.mkdir(FLAGS.data_dir)

    data_dir = os.path.join(FLAGS.data_dir, os.path.split(__file__.split(".")[0])[1])
    if os.path.isdir(data_dir) == False:
        os.mkdir(data_dir)


    CIFAR10_LINK = FLAGS.cifar10_link

    data_set_name = os.path.split(CIFAR10_LINK)[1]

    full_file_path = os.path.join(data_dir, data_set_name)

    if os.path.isfile(full_file_path) == False:
        print("downloading from: {}".format(CIFAR10_LINK))
        with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=CIFAR10_LINK.split("/")[-1]) as t:
            urllib.request.urlretrieve(CIFAR10_LINK, filename=full_file_path, reporthook=my_hook(t), data=None)
        print("finish download")



    return None


def main(argv):
    """

    :param argv:
    :return:
    """
    get_data()

if __name__ == '__main__':
    app.run(main)

