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
import tarfile

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


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='bytes')

    return dict


def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))

    out[range(n), vec] = 1
    return out


class CifarLoader(object):

    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d[b"data"] for d in data])
        n = len(images)

        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255.
        self.labels = one_hot(np.hstack([d[b"labels"] for d in data]), 10)

        return self

    def nex_batch(self, batch_size):
        x, y = self.images[self._i:self._i + batch_size], self.labels[self._i:self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.images)

        return x, y


class CifarDataManager(object):

    def __init__(self, data_dir):
        self.train = CifarLoader(["{}/data_batch_{}".format(data_dir, i) for i in range(1, 6)]).load()
        self.test = CifarLoader(["{}/test_batch".format(data_dir)]).load()


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

    data_dir = os.path.split(full_file_path)[0]
    tar = tarfile.open(full_file_path)
    folder_name = tar.getnames()[0]
    tar.extractall(path=data_dir)
    tar.close()

    return data_dir + "/" + folder_name


def CNN_base(input_tf, hold_prob, is_training=True):
    """

    :param input_tf:
    :param hold_prob:
    :param is_training:
    :return:
    """
    endpoints = {}
    out = tf.layers.Conv2D(filters=32, kernel_size=5, padding="SAME", activation=tf.nn.relu)(input_tf)
    endpoints[out.name] = out

    out = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME")(out)
    endpoints[out.name] = out

    out = tf.layers.Conv2D(filters=64, kernel_size=5, padding="SAME", activation=tf.nn.relu)(out)
    endpoints[out.name] = out

    out = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME")(out)
    endpoints[out.name] = out

    out = tf.layers.Flatten()(out)
    endpoints[out.name] = out

    out = tf.layers.Dense(units=1024, activation=tf.nn.relu)(out)
    endpoints[out.name] = out

    out = tf.layers.Dropout(rate=hold_prob)(out)
    endpoints[out.name] = out

    out = tf.layers.Dense(units=10)(out)
    endpoints[out.name] = out

    return out, endpoints


def main(argv):
    """

    :param argv:
    :return:
    """
    data_dir = get_data()

    d = CifarDataManager(data_dir)
    train_images = d.train.images
    train_labels = d.train.labels

    test_images = d.test.images
    test_labels = d.test.labels

    # display_cifar(images, 10)

    WIDTH, HEIGHT, CHANEL = train_images[0].shape
    NUM_CLASSES = train_labels.shape[1]

    BATCH_SIZE = 64
    TOTAL_BATCH = train_labels.shape[0] // BATCH_SIZE

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")
    hold_prob = tf.placeholder(tf.float32, name="hold_prob")

    # model
    y_pred, endpoints = CNN_base(input_tf=x, hold_prob=hold_prob, is_training=True)

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
    for key in temp_key[0:4]:
        tf.summary.image(key, tf.expand_dims(endpoints[key][:, :, :, 0], axis=3), FLAGS.batch_size)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        file_train_writer = tf.summary.FileWriter(FLAGS.model_dir + "/train", sess.graph)
        file_test_writer = tf.summary.FileWriter(FLAGS.model_dir + "/test")

        test_images = d.test.images
        test_labels = d.test.labels
        batch_num = int(len(d.train.images) / BATCH_SIZE)

        for epoch_id in range(FLAGS.epochs):
            for batch_id in range(batch_num):
                batch = d.train.nex_batch(FLAGS.batch_size)
                sess.run(train_op, feed_dict={x: batch[0],
                                              y_true: batch[1],
                                              hold_prob: 0.5})

            # PRINT OUT A MESSAGE EVERY 100 STEPS
            if epoch_id % 10 == 0:
                train_merged_info = sess.run(merged, feed_dict={x: batch[0],
                                                                y_true: batch[1],
                                                                hold_prob: 0.5})

                X = d.test.images.reshape(100, 100, 32, 32, 3)
                Y = d.test.labels.reshape(100, 100, 10)
                np_acc = np.mean([sess.run(acc, feed_dict={x: X[i],
                                                           y_true: Y[i],
                                                           hold_prob: 1.0}) for i in
                                  range(100)])

                test_merged_info = np.mean([sess.run(merged, feed_dict={x: X[i],
                                                                        y_true: Y[i],
                                                                        hold_prob: 1.0}) for i in
                                            range(100)])

                print('Step {}: accuracy={}'.format(epoch_id, np_acc))

                file_train_writer.add_summary(train_merged_info, epoch_id)
                file_test_writer.add_summary(test_merged_info, epoch_id)

                saver.save(sess, save_path=FLAGS.model_dir + "/" + os.path.split(__file__.split(".")[0])[1],
                           global_step=epoch_id)

        saver.save(sess, save_path=FLAGS.model_dir + "/" + os.path.split(__file__.split(".")[0])[1],
                   global_step=epoch_id)


if __name__ == '__main__':
    app.run(main)
