#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016


# CUDA_VISIBLE_DEVICES=""
import zz_ydwu_test_2.create_dataset as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim


if __name__ == "__main__":
    argv = ['--dataset_dir','/home/ydwu/work/facenet/zz_ydwu_test_2/origine_dataset-3000',#-2#-3000#-3002
            '--model','/home/ydwu/work/facenet/origine-models/facenet_bake/20170512-110547.pb',
            '--image_size', '160',
            '--margin', '44',
            '--save_dir', '/home/ydwu/work/facenet/zz_ydwu_test_2/face_emb_dataset-3000/face_emb_',#face_emb_dataset-3002/face_emb_', #face_emb_dataset-2/face_emb_',
            '--gpu_memory_fraction', '0.8']
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)
