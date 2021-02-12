import numpy as np
import tensorflow as tf
from onnx_tf.common import sys_config


class PadMixin(object):

  @classmethod
  def get_padding_as_op(cls, x, pads):
    num_dim = int(len(pads) / 2)

    tf_pads = np.transpose(np.array(pads).reshape([2, num_dim]))

    if sys_config.device == 'MCU':
        # make sure to pad the right dims in channel last tensor format
        tf_pads = [0, 0] + tf_pads.flatten().tolist() + [0, 0]
    else:
        tf_pads = [0, 0, 0, 0] + tf_pads.flatten().tolist()

    padding = tf.constant(
        np.array(tf_pads).reshape([num_dim + 2, 2])
        .astype(np.int32))  # tf requires int32 paddings
    return tf.pad(x, padding)
