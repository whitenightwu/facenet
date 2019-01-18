#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016


# CUDA_VISIBLE_DEVICES=""
import src.validate_on_lfw as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim

#########
if __name__ == "__main__":
    argv = ['--lfw_dir', '/home/ydwu/datasets/white-lfw',
            '--model', '/home/ydwu/models/tmp',#train_nri_mobilenet/tmp',#origine-models/20170512-110547', #'/home/ydwu/work/facenet/results_models/tripletloss_resnet/white-models-inception_resnet_v1/20180809-173822',#'/home/ydwu/work/facenet/origine-models/facenet_bake/20170512-110547.pb',
            '--lfw_batch_size', '100',
            '--lfw_pairs','/home/ydwu/work/facenet/data/pairs.txt',
            '--lfw_nrof_folds', '10',
            '--distance_metric', '0',        
    ]
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)
