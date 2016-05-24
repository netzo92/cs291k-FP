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


def eval_once(eval_data, model_path, global_step, saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    saver.restore(sess, model_path) //load variables to saved model
    coord = tf.train.Coordinator() #create coordinator object, used to stop and join on all threads
    try:
      threads = []  #init initial empty list used for threads
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):  #Add all current queue runners to this list.
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True)) #Start and append these queue runners
      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])  #Compute which predictions were correct.
        true_count += np.sum(predictions)   #Add up the total number of good guesses
        step += 1

      
      precision = true_count / total_sample_count # Compute performance accuracy
      if eval_data is 'test':
        print('Testing Accuracy:  %.3f' % (precision))
      elif eval_data is 'train':
        print('Training Accuracy:  %.3f'%(precision))
      elif eval_data is 'val':
        print('Validation Accuracy:  %.3f'%(precision))


      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop() #Using the coordinator to request_stop from threads
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(eval_data, model_path, global_step ):
  """Eval CIFAR-100 prediction performance."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-100
    images, labels = data_utils.inputs(eval_data=eval_data, data_dir = FLAGS.data_dir, batch_size=FLAGS.batch_size) #Get batches

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = conv_net.inference(images)   #Run predictions on the images
    logits_norm = tf.nn.softmax(logits)   #Check the softmax of the images, this should normalize our scores for predictions
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits_norm, labels, 1) #Get the highest ranked logit_norms

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        conv_net.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    eval_once(eval_data, model_path, global_step, saver, summary_writer, top_k_op, summary_op)


def choose_model():
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir) #Use this for opening the model that we just trained.
    if ckpt and ckpt.model_checkpoint_path:
      print('Using model located at: '+ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      return ckpt.model_checkpoint_path, global_step
    else:
      print('No checkpoint file found exiting')
      return None, None

def main(argv=sys.argv):
  if len(sys.argv) > 1:
    FLAGS.data_dir = sys.argv[1]
  conv_net.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  conv_net.main()
  model_path, global_step = choose_model()
  if model_path is not None:
    evaluate('train', model_path, global_step)  #
    evaluate('val', model_path, global_step)
    evaluate('test', model_path, global_step)

if __name__ == '__main__':
  tf.app.run()
