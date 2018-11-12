#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016


# CUDA_VISIBLE_DEVICES=""
import src.freeze_mtcnn as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim

#########
if __name__ == "__main__":
    argv = []
    
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)
