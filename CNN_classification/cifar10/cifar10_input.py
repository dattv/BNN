from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import tensorflow_datasets as tfd

import cifar10.cifar10_config as config

def _get_images_labels(batch_size, split, distords=False):
    """

    :param batch_size:
    :param split:
    :param distords:
    :return:
    """

    dataset = tfd.load(name=config.DATASET_NAME, split=split)

    scope = "input"
    if distords:
        scope = "data_augmentation"

    with tf.name_scope(scope) as sc:
        dataset = dataset.map(DataPreprocessor(distords), num_parallel_calls=config.NUM_PARALEL_CALLS)

    dataset = dataset.prefetch(-1)
    dataset = dataset.repeat().batch(batch_size)
    iterator = dataset.make_one_short_iterator()
    images_labels = iterator.get_next()
    images, labels = images_labels['input'], images_labels['target']
    tf.summary.image('images', images)
    return images, labels


class DataPreprocessor(object):
    def __init__(self, distord):

        self._distord = distord

    def __call__(self, record):

        img = record["image"]
        img = tf.cast(img, tf.float32)

        if self._distord:
            img = tf.random_crop(img, [config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS])
            img = tf.image.random_flip_left_right(img)

            img = tf.image.random_brightness(img, max_delta=63)
            img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        else:
            img = tf.image.resize_image_with_crop_or_pad(img, config.IMAGE_SIZE, config.IMAGE_SIZE)

        img = tf.image.per_image_standardization(img)

        return dict(input=img, target=record['label'])

def distored_input(batch_size):
    """

    :param batch_size:
    :return:
    """
    return _get_images_labels(batch_size=batch_size, split=tfd.Split.TRAIN, distords=True)


def inputs(eval_data, batch_size):
    """

    :param eval_data:
    :param batch_size:
    :return:
    """
    split = tfd.Split.TRAIN
    if eval_data == "test":
        split = tfd.Split.TEST

    return _get_images_labels(batch_size=batch_size, split=split)



