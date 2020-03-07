from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
import cifar10.cifar10_input

from absl import app, flags
import cifar10.cifar10_config as config

flags.DEFINE_string('batch_size',
                    default=config.BATCH_SIZE,
                    help="batch size")

flags.DEFINE_bool('use_fp16',
                  default=False,
                  help='use fp16 or not')
flags.DEFINE_integer('image_size',
                     default=config.IMAGE_SIZE,
                     help='size of image')
flags.DEFINE_integer('image_channels',
                     default=config.IMAGE_CHANNELS,
                     help="image channels number")

FLAGS = flags.FLAGS


def _activation_summary(x):
    """

    :param x:
    :return:
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % config.TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
