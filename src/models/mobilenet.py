#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : mobilenet.py
## Authors    : ydwu@ydwu-OptiPlex-5050
## Create Time: 2018-07-20:14:15:58
## Description:
## 
##
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):


  logits, end_points = mobilenet(images, num_classes=bottleneck_layer_size, is_training=phase_train, weight_decay=weight_decay)
  return logits, end_points



def mobilenet(inputs,
              num_classes=1000,
              is_training=True,
              width_multiplier=1,
              scope='MobileNet',
              weight_decay=5e-4):
  """ MobileNet
  More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """

  def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                width_multiplier,
                                sc,
                                downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  scope=sc+'/depthwise_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pointwise_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
    return bn




  #############################################################################################
  #################################
  
  with tf.variable_scope(scope) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=slim.xavier_initializer_conv2d(),
                        # weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                        # weights_init = tf.truncated_normal_initializer(stddev=0.09)
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=None,
                        biases_initializer=slim.init_ops.zeros_initializer(),
                        outputs_collections=[end_points_collection]):
                        # normalizer_fn=slim.batch_norm):
      with slim.arg_scope([slim.batch_norm],
                          is_training=True,
                          activation_fn=tf.nn.relu,
                          updates_collections=None,
                          decay=0.995,
                          zero_debias_moving_mean=True,
                          epsilon=0.001,
                          variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                          center=True,
                          scale=True,
                          fused=True):

  #################################
  # with tf.variable_scope(scope) as sc:
  #   end_points_collection = sc.name + '_end_points'
  #   with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
  #                       weights_initializer=slim.initializers.xavier_initializer(),
  #                       weights_regularizer=slim.l2_regularizer(weight_decay),
  #                       activation_fn=None,
  #                       biases_initializer=slim.init_ops.zeros_initializer(),
  #                       outputs_collections=[end_points_collection]):
  #     with slim.arg_scope([slim.batch_norm],
  #                         is_training=True,
  #                         activation_fn=tf.nn.relu,
  #                         updates_collections=None,
  #                         decay=0.995,
  #                         zero_debias_moving_mean=True,
  #                         epsilon=0.001,
  #                         variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
  #                         center=True,
  #                         scale=True,
  #                         fused=True):
        
        net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
        print(net)

        # net = slim.batch_norm(net, scope='conv_1/batch_norm')
        net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
        print(net)
    
        net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
        net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
        print(net)

        net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
        net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
        print(net)

        net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')
        print(net)

        net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
        net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
        print(net)

        net = slim.avg_pool2d(net, [5, 5], scope='avg_pool_15')
        print(net)


        #         batch_norm_params = {
        #     # Decay for the moving averages
        #     'decay': 0.995,
        #     # epsilon to prevent 0s in variance
        #     'epsilon': 0.001,
        #     # force in-place updates of mean and variance estimates
        #     'updates_collections': None,
        #     # Moving averages ends up in the trainable variables collection
        #     'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
        #     # Only update statistics during training mode
        #     'is_training': phase_train_placeholder
        # }
    end_points = slim.utils.convert_collection_to_dict(end_points_collection)


    with slim.arg_scope([slim.batch_norm],
                        is_training=True,
                        activation_fn=None,
                        updates_collections=None,
                        decay=0.995,
                        zero_debias_moving_mean=True,
                        epsilon=0.001,
                        variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                        center=False,
                        scale=False,
                        fused=True):

      logits = slim.conv2d(net,
                           num_classes,
                           [1, 1],
                           activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                           # weights_regularizer=1#slim.l2_regularizer(1),
                           normalizer_fn=None,#slim.batch_norm,
                           biases_initializer=None,
                           scope='Conv2d_1c_1x1')
      print(logits)

    # logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
    logits = tf.reshape(logits, [-1, num_classes], name='Reshape')          
    print(logits)

    end_points['Logits'] = logits

  return logits, end_points
