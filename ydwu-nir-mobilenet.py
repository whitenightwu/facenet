#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016


# CUDA_VISIBLE_DEVICES=""
import src.train_tripletloss as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim

if __name__ == "__main__":
    argv = ['--logs_base_dir','/home/ydwu/work/facenet/train_nri_mobilenet',
            '--models_base_dir','/home/ydwu/work/facenet/train_nri_mobilenet',
            '--data_dir', '/media/ydwu/Office/white-ms1mclean-black',
            
            # '--pretrained_model', '/home/ydwu/work/facenet/train_office_mobilenet/20180928-093249/model-20180928-093249.ckpt-186752',
            
            '--model_def', 'src.models.mobilenet',
            '--gpu_memory_fraction','0.8',
            '--image_size','160',
            # '--random_flip',
            '--random_crop',
            '--seed','995',
            '--epoch_size', '600',
            '--max_nrof_epochs', '1000',
            '--batch_size', '120',#'36',#'120',
            '--people_per_batch', '256',#'32',#'256',
            '--images_per_person', '6',
            '--learning_rate', '0.5',#'0.05',#0.1 # 0.5
            '--alpha', '0.2',#'0.4', 
            '--weight_decay', '5e-4',#'1e-4',
            '--moving_average_decay', '0.9999',
            '--optimizer', 'ADAGRAD', #'RMSPROP',#'ADAM','ADAGRAD'
            '--learning_rate_decay_factor', '1.0',
            '--lfw_pairs','/home/ydwu/work/facenet/data/pairs.txt',
            '--lfw_dir', '/home/ydwu/datasets/white-lfw-black',
            '--lfw_nrof_folds', '2' ,
            '--embedding_size', '128']
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)
