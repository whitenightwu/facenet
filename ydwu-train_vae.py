#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016

import src.generative.train_vae as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim
 
if __name__ == "__main__":
    argv = ['--models_base_dir','/home/ydwu/work/facenet/src/generative/vae_result',
            '--data_dir','/media/ydwu/Document/Datasets/lfw_160',
            '--save_every_n_steps', '10',
            '--batch_size', '128',

            
            '--vae_def', 'src.generative.models.dfc_vae',
            '--reconstruction_loss_type', 'PLAIN']
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)

# '--data_dir','/media/ydwu/Document/Datasets/white-ms1mclean',
# '--model_def', 'src.generative.models.dfc_vae',
# '--model_def', 'src.models.inception_resnet_v1',
