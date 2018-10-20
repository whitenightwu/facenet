#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016


# CUDA_VISIBLE_DEVICES=""
# import zz_ydwu_test_2.face_recognition as facenet_train
import zz_ydwu_face_search.face_recognition_bak as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim


######### result_dataset_v2
if __name__ == "__main__":
    argv = ['--dataset_dir','/home/ydwu/datasets/face_near_infrared_test _v3',#'/home/ydwu/work/facenet/zz_ydwu_face_search/test_dataset_align-v2',#'/home/ydwu/work/facenet/zz_ydwu_test_2/origine_dataset-v2-05',
            '--dataset_file', '/tmp/ydwu-face/face_emb_0',#'/home/ydwu/work/facenet/zz_ydwu_test_2/face_emb_dataset-3002/face_emb_0',#face_emb',
            '--model','/home/ydwu/work/facenet/origine-models/facenet_bake/20170512-110547.pb',
            '--image_size', '160',
            '--margin', '12',
            '--gpu_memory_fraction', '0.8']
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)
