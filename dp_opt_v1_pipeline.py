# Copyright 2020, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Vectorized differentially private optimizers for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow.compat.v1 as tf

AdagradOptimizer = tf.train.AdagradOptimizer
AdamOptimizer = tf.train.AdamOptimizer
GradientDescentOptimizer = tf.train.GradientDescentOptimizer

parent_code = tf.train.Optimizer.compute_gradients.__code__
GATE_OP = tf.train.Optimizer.GATE_OP  # pylint: disable=invalid-name
import re

from tensorflow.python.ops import gen_math_ops

from tensorflow.python.ops import math_ops

from tensorflow.python.framework import ops



def make_vectorized_optimizer_class(cls):
  """Constructs a vectorized DP optimizer class from an existing one."""
  child_code = cls.compute_gradients.__code__
  if child_code is not parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class DPOptimizerClass(cls):
    """Differentially private subclass of given class cls."""

    def __init__(
        self,
        l2_norm_clip,
        noise_std, effective_batch_size, #use_pipeline,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the DPOptimizerClass.

      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients)
        noise_multiplier: Ratio of the standard deviation to the clipping norm
        num_microbatches: How many microbatches into which the minibatch is
          split. If None, will default to the size of the minibatch, and
          per-example gradients will be computed.
      """
      super(DPOptimizerClass, self).__init__(*args, **kwargs)
      self._l2_norm_clip = l2_norm_clip
      self._effective_batch_size = effective_batch_size
      self._noise_std = noise_std * self._l2_norm_clip/np.sqrt(self._effective_batch_size),
    def compute_gradients(self,
                          loss,
                          var_list,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=True,
                          grad_loss=None,
                          gradient_tape=None):
      if callable(loss):
        # TF is running in Eager mode
        raise NotImplementedError('Vectorized optimizer unavailable for TF2.')
      else:
        # TF is running in graph mode, check we did not receive a gradient tape.
        if gradient_tape:
          raise ValueError('When in graph mode, a tape should not be passed.')


        self._num_microbatches = 1
        microbatch_losses = tf.reshape(loss, [self._num_microbatches, -1])
        if var_list is None:
          var_list = (
              tf.trainable_variables() + tf.get_collection(
                  tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        grads, var_list = zip(*super(DPOptimizerClass, self).compute_gradients(
              microbatch_losses,
              var_list,
              gate_gradients,
              aggregation_method,
              colocate_gradients_with_ops,
              grad_loss))
        grads_list = [
              g if g is not None else tf.zeros_like(v)
              for (g, v) in zip(list(grads), var_list)
          ]


        pipeline_stages = []
        _vars = []
        grads_pipeline_stages = []
        pipeline_stages_to_grads = {}
        for grad, var in zip(grads_list, var_list):
            _vars.append(var)
            m = re.match(r'.*(Pipeline_stage_[0-9]*).*', grad.op.name)
            if m:
                pipeline_stage = m.group(1)
                grads_pipeline_stages.append(pipeline_stage)
                if not pipeline_stage in pipeline_stages:
                    pipeline_stages.append(pipeline_stage)
                if not pipeline_stage in pipeline_stages_to_grads:
                    pipeline_stages_to_grads[pipeline_stage] = []
                pipeline_stages_to_grads[pipeline_stage].append(grad)
        normalized_grads = []
        for pipeline_stage in pipeline_stages:
            with ops.get_default_graph().colocate_with(pipeline_stages_to_grads[pipeline_stage][0].op):
                squared_l2_norms = [
                    math_ops.reduce_sum(input_tensor=gen_math_ops.square(g)) for g in pipeline_stages_to_grads[pipeline_stage]
                ]
                pipeline_stage_norm = math_ops.sqrt(math_ops.add_n(squared_l2_norms))
                for grad in pipeline_stages_to_grads[pipeline_stage]:
                    div = tf.maximum(pipeline_stage_norm / self._l2_norm_clip, 1.)
                    grad = grad / div

                    noise = tf.random.normal(tf.shape(input=grad), stddev=self._noise_std, dtype=tf.float32)
                    noise = tf.cast(noise, tf.float16)
                    grad += noise
                    normalized_grads.append(grad)
        grads_and_vars = list(zip(normalized_grads, _vars))
        return grads_and_vars

  return DPOptimizerClass


VectorizedDPAdagrad = make_vectorized_optimizer_class(AdagradOptimizer)
VectorizedDPAdam = make_vectorized_optimizer_class(AdamOptimizer)
VectorizedDPSGD = make_vectorized_optimizer_class(GradientDescentOptimizer)
