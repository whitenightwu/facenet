#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016

import src.calculate_filtering_metrics as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim
 
if __name__ == "__main__":
    argv = ['--dataset_dir','/home/ydwu/work/facenet/ydwu-test_1/dataset',#'/home/ydwu/datasets/white-lfw',
            '--model_file','/home/ydwu/work/facenet/origine-models/facenet_bake/20170512-110547.pb',
            '--data_file_name','/home/ydwu/work/facenet/ydwu-test_1/filtering_metrics',
            '--image_size','160',
            '--batch_size', '70']
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)

