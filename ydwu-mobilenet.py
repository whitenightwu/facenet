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
    argv = ['--logs_base_dir','/home/ydwu/work/facenet/train_office_mobilenet',
            '--models_base_dir','/home/ydwu/work/facenet/train_office_mobilenet',
            '--data_dir', '/media/ydwu/Document/Datasets/white-ms1mclean', #'/media/ydwu/Office/tmp_dataset',
            
            '--pretrained_model', '/home/ydwu/work/facenet/train_office_mobilenet/20180927-175921/model-20180927-175921.ckpt-26036',#20180925-131347/model-20180925-131347.ckpt-203166',#20180921-093412/model-20180921-093412.ckpt-33821',#20180919-102235/model-20180919-102235.ckpt-116049',#20180918-163310/model-20180918-163310.ckpt-68368',#20180918-101807/model-20180918-101807.ckpt-20608',#20180917-114021/model-20180917-114021.ckpt-87983',#20180912-194025/model-20180912-194025.ckpt-246230',
            #train_office_mobilenet/20180910-093513/model-20180910-093513.ckpt-225303',#20180906-132015/model-20180906-132015.ckpt-175994',#20180903-091340/model-20180903-091340.ckpt-240521',#work/facenet/train_office_mobilenet/20180831-161429/model-20180831-161429.ckpt-83882',#20180830-093952/model-20180830-093952.ckpt-25896
            '--model_def', 'src.models.mobilenet',#JZ_#.office
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
            '--learning_rate', '0.01',#'0.05',#0.1 # 0.5
            '--alpha', '0.2',#'0.4', 
            '--weight_decay', '5e-4',#'1e-4',
            '--moving_average_decay', '0.9999',
            '--optimizer', 'ADAGRAD', #'RMSPROP',#'ADAM','ADAGRAD'
            '--learning_rate_decay_factor', '1.0',
            '--lfw_pairs','/home/ydwu/work/facenet/data/pairs.txt',
            '--lfw_dir', '/home/ydwu/datasets/white-lfw',
            '--lfw_nrof_folds', '2' ,
            '--embedding_size', '128']
    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)

    # learning_rate_decay_factor: 1.0

    # '--keep_probability', '0.7', #0.7
    # '--data_dir','/media/ydwu/Document/Datasets/white-ms1mclean',
    # '--pretrained_model', '/home/ydwu/work/facenet/train_office_mobilenet/20180823-140011/model-20180823-140011.ckpt-14660',
            
    # '--learning_rate', '-1',
    # '--learning_rate_schedule_file', '/home/ydwu/work/facenet/data/learning_rate_retrain_tripletloss.txt',
    
    # --learning_rate_decay_epochs
    # --learning_rate_decay_factor
