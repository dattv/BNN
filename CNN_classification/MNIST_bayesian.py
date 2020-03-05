from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
                         'MNIST_BNN/'),
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


def BNN_MNIST(input_tf, hold_prob, is_training=True):
    """

    :param input_tf:
    :param hold_prob:
    :param is_training:
    :return:
    """
    endpoints = {}
    input_reshape = tf.reshape(input_tf, shape=[-1, 28, 28, 1], name="input_reshape")
    endpoints[input_reshape.name] = input_reshape

    out = tfp.layers.Convolution2DReparameterization(filters=32,
                                                     kernel_size=5,
                                                     padding="SAME",
                                                     activation=tf.nn.relu)(input_reshape)
    endpoints[out.name] = out

    out = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME")(out)
    endpoints[out.name] = out

    out = tfp.layers.Convolution2DReparameterization(filters=64,
                                                     kernel_size=5,
                                                     padding="SAME",
                                                     activation=tf.nn.relu)(out)
    endpoints[out.name] = out

    out = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME")(out)
    endpoints[out.name] = out

    out = tf.layers.Flatten()(out)
    endpoints[out.name] = out

    out = tfp.layers.DenseFlipout(units=1024, activation=tf.nn.relu)(out)
    endpoints[out.name] = out

    out = tf.layers.Dropout(rate=hold_prob)(out)
    endpoints[out.name] = out

    out = tfp.layers.DenseFlipout(units=10)(out)
    endpoints[out.name] = out

    return out, endpoints


def main(argv):
    """

    :param argv:
    :return:
    """
    mnist_train, mnist_test = get_data()

    #
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")
    hold_prob = tf.placeholder(tf.float32, name="hold_prob")

    # model
    y_pred, endpoints = BNN_MNIST(input_tf=x, hold_prob=hold_prob, is_training=True)

    # loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred,
                                                   labels=y_true)
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=1.e-3)
    train_op = optimizer.minimize(cross_entropy)

    matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

    acc_summary = tf.summary.scalar(tensor=acc, name="acc_summary")
    loss_summary = tf.summary.scalar(tensor=cross_entropy, name="loss_cross_entropy")
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)

    temp_key = [key for key in endpoints.keys()]
    for key in temp_key[0:5]:
        tf.summary.image(key, tf.expand_dims(endpoints[key][:, :, :, 0], axis=3), FLAGS.batch_size)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        file_train_writer = tf.summary.FileWriter(FLAGS.model_dir + "/train", sess.graph)
        file_test_writer = tf.summary.FileWriter(FLAGS.model_dir + "/test")

        for epoch_id in range(FLAGS.epochs):
            batch_x, batch_y = mnist_train.next_batch(FLAGS.batch_size)
            sess.run(train_op, feed_dict={x: batch_x,
                                          y_true: batch_y,
                                          hold_prob: 0.5})

            # PRINT OUT A MESSAGE EVERY 100 STEPS
            if epoch_id % 500 == 0:
                train_merged_info = sess.run(merged, feed_dict={x: batch_x,
                                                                y_true: batch_y,
                                                                hold_prob: 0.5})

                np_acc, test_merged_info = sess.run([acc, merged], feed_dict={x: mnist_test.images,
                                                                              y_true: mnist_test.labels,
                                                                              hold_prob: 0.5})
                print('Step {}: accuracy={}'.format(epoch_id, np_acc))

                file_train_writer.add_summary(train_merged_info, epoch_id)
                file_test_writer.add_summary(test_merged_info, epoch_id)

                saver.save(sess, save_path=FLAGS.model_dir + "/" + os.path.split(__file__.split(".")[0])[1],
                           global_step=epoch_id)

        saver.save(sess, save_path=FLAGS.model_dir + "/" + os.path.split(__file__.split(".")[0])[1],
                   global_step=epoch_id)


if __name__ == '__main__':
    app.run(main)
