# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# code adapted from https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/models/image/cifar10 by Metehan Ozten
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math, time, os, sys

import numpy as np
import tensorflow as tf
import conv_net
import data_utils
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', os.path.join(os.getcwd(),'cifar10_eval'),
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', os.path.join(os.getcwd(),'cifar10_train'),
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")


def main(argv=sys.argv):
  if len(sys.argv) > 1:
    FLAGS.data_dir = sys.argv[1]
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  conv_net.main()


if __name__ == '__main__':
  tf.app.run()
