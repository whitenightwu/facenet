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
    argv = ['--model', '/home/ydwu/work/facenet/origine-models/20170512-110547',

            '--lfw_dir', '/home/ydwu/datasets/face_near_infrared_test/black',
            # '--lfw_dir', '/home/ydwu/datasets/face_near_infrared_test/RGB',
            
            # '--lfw_dir', '/home/ydwu/datasets/face_near_infrared_test/RGB_fake',
            # '--lfw_dir', '/home/ydwu/datasets/face_near_infrared_test/black_fake',

            '--lfw_batch_size', '99',#'50',#'99',
            
            '--lfw_pairs','/home/ydwu/work/facenet/nir2.txt',
            '--lfw_nrof_folds', '10',
            '--distance_metric', '0',        
    ]
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)
