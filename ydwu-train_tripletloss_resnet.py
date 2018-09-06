#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016

import src.train_tripletloss as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim
 
if __name__ == "__main__":
    argv = ['--logs_base_dir','/home/ydwu/work/facenet/white-logs-inception_resnet_v1',
            '--models_base_dir','/home/ydwu/work/facenet/white-models-inception_resnet_v1',
            '--data_dir','/media/ydwu/Document/Datasets/white-ms1mclean',
            '--model_def', 'src.models.inception_resnet_v1',
            '--gpu_memory_fraction','0.8',
            '--alpha','0.40', 
            '--seed','995',
            '--epoch_size', '900',
            '--max_nrof_epochs', '1000',
            '--batch_size','60', #90
            '--people_per_batch','15', #12 
            '--images_per_person', '9',
            '--image_size','160',
            '--learning_rate', '-1', #'0.0001',#'0.001',
            '--learning_rate_schedule_file', '/home/ydwu/work/facenet/data/learning_rate_retrain_tripletloss.txt',
            '--weight_decay', '1e-4',
            '--optimizer', 'RMSPROP',
            '--random_flip',
            '--random_crop',
            '--keep_probability', '0.5', #0.7
            '--lfw_pairs','/home/ydwu/work/facenet/data/pairs.txt',
            '--lfw_dir', '/home/ydwu/datasets/white-lfw',
            '--lfw_nrof_folds', '2' ,
            '--embedding_size', '128']
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)


            # '--pretrained_model', '/home/ydwu/work/facenet/white-models-inception_resnet_v1/20180803-165554/model-20180803-165554.ckpt-228020',
            # '--pretrained_model', '/home/ydwu/work/facenet/white-models-inception_resnet_v1/20180806-092115/model-20180806-092115.ckpt-391764',
            # '--pretrained_model', '/home/ydwu/work/facenet/white-models-inception_resnet_v1/20180809-091843/model-20180809-091843.ckpt-53593',



            # --alpha
            # --learning_rate_decay_epochs
            # --learning_rate_decay_factor
            # --moving_average_decay