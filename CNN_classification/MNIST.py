from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import os
import warnings
import logging
import matplotlib.pyplot as plt
from absl import flags
from absl import app

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
                         'bayesian_neural_network_MNIST/'),
    help="Directory to put the model's fit.")
flags.DEFINE_integer('viz_steps',
                     default=400,
                     help='Frequency at which save visualizations.')

flags.DEFINE_integer('epochs',
                     default=5000,
                     help='number of maximum epochs')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


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

    mnist_onehot = input_data.read_data_sets(data_dir, one_hot=True)
    mnist_train = mnist_onehot.train
    mnist_test = mnist_onehot.test

    return mnist_train, mnist_test

def CNN_base(input_tf, hold_prob, is_training=True):
    """

    :param input_tf:
    :param hold_prob:
    :param is_training:
    :return:
    """
    out = tf.layers.Conv2D(filters=32, kernel_size=5, padding="SAME", activation=tf.nn.relu)(input_tf)
    out = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME")(out)
    out = tf.layers.Conv2D(filters=64, kernel_size=5, padding="SAME", activation=tf.nn.relu)(out)
    out = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME")(out)
    out = tf.layers.Flatten()(out)
    out = tf.layers.Dense(units=1024, activation=tf.nn.relu)(out)
    out = tf.layers.Dropout(rate=hold_prob)(out)
    out = tf.layers.Dense(units=10)(out)
    return out


def main(argv):
    """

    :param argv:
    :return:
    """
    mnist_train, mnist_test = get_data()

    #
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])
    hold_prob = tf.placeholder(tf.float32)

    # model
    y_pred = CNN_base(input_tf=x, hold_prob=hold_prob, is_training=True)

    # loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred,
                                                   labels=y_true)
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=1.e-3)
    train_op = optimizer.minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())



if __name__ == '__main__':
    app.run(main)
