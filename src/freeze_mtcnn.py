#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : freeze_mtcnn.py
## Authors    : ydwu@ydwu-OptiPlex-5050
## Create Time: 2018-10-22:16:43:56
## Description:
## 
##

"""Performs face alignment and stores face thumbnails in the output directory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append("/home/ydwu/work/facenet/src")

import sys
import tensorflow as tf
import align.detect_face
import argparse
def main(args):
    print('Freezing the PNet, RNet and ONet models')
    
    # with tf.Graph().as_default():
    #     sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    #     with sess.as_default():
    #         align.detect_face.freeze_mtcnn(sess, None)
            
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)            
            align.detect_face.freeze('ydwu_mtcnn', sess, ['pnet/conv4-2/BiasAdd', 'pnet/prob1', 'rnet/conv5-2/conv5-2', 'rnet/prob1', 'onet/conv6-2/conv6-2', 'onet/conv6-3/conv6-3', 'onet/prob1'])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
