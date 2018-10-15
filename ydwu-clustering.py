#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016


# CUDA_VISIBLE_DEVICES=""
import contributed.clustering as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim

#########
if __name__ == "__main__":
    argv = ['--model_dir', '/home/ydwu/work/facenet/origine-models/20170512-110547',
            '--batch_size', '50',#'50',
            '--input', '/home/ydwu/datasets/face_near_infrared_test',
            '--output', '/tmp/face_near_infrared_test',

    ]
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)
