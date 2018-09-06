#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016


# CUDA_VISIBLE_DEVICES=""
import zz_ydwu_test_2.creat_embedding_compare as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim


# ######### result_dataset_v2
# if __name__ == "__main__":
#     argv = ['--model','/home/ydwu/work/facenet/origine-models/facenet_bake/20170512-110547.pb',
#             '--image_files',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset_v2/01/webwxgetmsgimg3_1.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset_v2/02/webwxgetmsgimg2_1.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset_v2/03/webwxgetmsgimg2_2.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset_v2/hu/webwxgetmsgimg_0.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset_v2/hu/webwxgetmsgimg2_0.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset_v2/hu/webwxgetmsgimg3_0.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset_v2/hu/webwxgetmsgimg4_0.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset_v2/xi/webwxgetmsgimg4_1.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset_v2/xi/webwxgetmsgimg5_0.png',
#             '--image_size', '160',
#             '--margin', '44',
#             '--save_dir', '/home/ydwu/work/facenet/zz_ydwu_test_2/face_emb',
#             '--gpu_memory_fraction', '0.7']
#     print(argv)
#     print("-----src.facenet_train_sun--------------")
#     args = facenet_train.parse_arguments(argv)
#     facenet_train.main(args)




######### result_dataset_v2
if __name__ == "__main__":
    argv = ['--dataset_dir','/home/ydwu/work/facenet/zz_ydwu_test_2/origine_dataset',
            '--model','/home/ydwu/work/facenet/origine-models/facenet_bake/20170512-110547.pb',
            '--image_size', '160',
            '--margin', '44',
            '--save_dir', '/home/ydwu/work/facenet/zz_ydwu_test_2/face_emb',
            '--gpu_memory_fraction', '0.7']
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)
