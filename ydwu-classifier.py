#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016


# CUDA_VISIBLE_DEVICES=""
import src.classifier as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim

if __name__ == "__main__":
    argv = ['--mode','CLASSIFY',
            '--data_dir', '/home/ydwu/work/facenet/ydwu-test_1/dataset',
            '--model','/home/ydwu/work/facenet/origine-models/facenet_bake/20170512-110547.pb',
            '--classifier_filename', '01',
            '--batch_size', '9',
            '--image_size', '160']
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)
