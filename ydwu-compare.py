#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016


# CUDA_VISIBLE_DEVICES=""
import src.compare as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim

# ##############################################################
# ######### result_dataset
# if __name__ == "__main__":
#     argv = ['--model','/home/ydwu/work/facenet/origine-models/facenet_bake/20170512-110547.pb',
#             '--image_files',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset/01/webwxgetmsgimg2_1.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset/01/webwxgetmsgimg2_2.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset/01/webwxgetmsgimg3_1.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset/hu/webwxgetmsgimg_0.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset/hu/webwxgetmsgimg2_0.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset/hu/webwxgetmsgimg3_0.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset/hu/webwxgetmsgimg4_0.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset/xi/webwxgetmsgimg4_1.png',
#             '/home/ydwu/work/facenet/ydwu-test_1/result_dataset/xi/webwxgetmsgimg5_0.png',
#             '--image_size', '160',
#             '--margin', '44',
#             '--gpu_memory_fraction', '0.7']
#     print(argv)
#     print("-----src.facenet_train_sun--------------")
#     args = facenet_train.parse_arguments(argv)
#     facenet_train.main(args)




# ##############################################################
# ######### result_dataset_v2
# if __name__ == "__main__":
#     argv = ['--model','/home/ydwu/work/facenet/origine-models/facenet_bake/20170512-110547.pb',
#             '--image_files',

#             '/home/ydwu/work/facenet/zz_ydwu_face_search/dataset_and_result/test_dataset_align/01/webwxgetmsgimg2_0.png',
#             '/home/ydwu/work/facenet/zz_ydwu_face_search/dataset_and_result/test_dataset_align/01/webwxgetmsgimg2_1.png',
#             '/home/ydwu/work/facenet/zz_ydwu_face_search/dataset_and_result/test_dataset_align/01/webwxgetmsgimg2_2.png',

#             '/home/ydwu/work/facenet/zz_ydwu_face_search/dataset_and_result/test_dataset_align/01/webwxgetmsgimg3_0.png',
#             '/home/ydwu/work/facenet/zz_ydwu_face_search/dataset_and_result/test_dataset_align/01/webwxgetmsgimg3_1.png',
            
#             '/home/ydwu/work/facenet/zz_ydwu_face_search/dataset_and_result/test_dataset_align/01/webwxgetmsgimg4_0.png',
#             '/home/ydwu/work/facenet/zz_ydwu_face_search/dataset_and_result/test_dataset_align/01/webwxgetmsgimg4_1.png',
            
#             # '/home/ydwu/work/facenet/zz_ydwu_face_search/dataset_and_result/test_dataset_align/01/hu/webwxgetmsgimg_0.png',
#             # '/home/ydwu/work/facenet/zz_ydwu_face_search/dataset_and_result/test_dataset_align/01/hu/webwxgetmsgimg3_0.png',

#             # '/home/ydwu/work/facenet/zz_ydwu_face_search/dataset_and_result/test_dataset_align/01/xi/webwxgetmsgimg5_0.png',
            
#             '--image_size', '160',
#             '--margin', '44',
#             '--gpu_memory_fraction', '0.7']
#     print(argv)
#     print("-----src.facenet_train_sun--------------")
#     args = facenet_train.parse_arguments(argv)
#     facenet_train.main(args)


##############################################################
#########

if __name__ == "__main__":
    argv = ['--model','/home/ydwu/work/facenet/origine-models/facenet_bake/20170512-110547.pb',
            '--image_files',
            # '/home/ydwu/datasets/face_near_infrared/test_v1/54.png',
            # '/home/ydwu/datasets/face_near_infrared/test_v1/54-1.png',
            # '/home/ydwu/datasets/face_near_infrared/black-5/571.jpg',
            '/home/ydwu/datasets/face_near_infrared/color-5/571.jpg',
            # '/home/ydwu/datasets/face_near_infrared/black-6/1742.jpg',
            '/home/ydwu/datasets/face_near_infrared/color-6/1742.jpg',
            '/home/ydwu/datasets/face_near_infrared/black-7/21886.jpg',
            '/home/ydwu/datasets/face_near_infrared/color-7/21886.jpg',


            '/home/ydwu/datasets/face_near_infrared_test/10075_fake_B.png',
            
            '--image_size', '160',
            '--margin', '44',
            '--gpu_memory_fraction', '0.7']
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)

