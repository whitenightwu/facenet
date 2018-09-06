#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : ydwu-train_centerloss.py
## Authors    : ydwu@ydwu-OptiPlex-5050
## Create Time: 2018-08-03:15:48:57
## Description:
## 
##

import src.train_softmax as facenet_train
import tensorflow as tf
import tensorflow.contrib.slim as slim
 
if __name__ == "__main__":
    argv = ['--logs_base_dir','/home/ydwu/work/facenet/white-logs-inception_resnet_v1',
            '--models_base_dir','/home/ydwu/work/facenet/white-models-inception_resnet_v1',
            '--data_dir','/media/ydwu/Document/Datasets/white-ms1mclean',
            '--model_def', 'src.models.inception_resnet_v1',
            
            '--image_size', '160',
            '--gpu_memory_fraction','0.8',
            '--embedding_size', '128',
            '--seed','995',
            '--random_flip',
            '--random_crop',
            '--keep_probability', '0.8',

            '--use_fixed_image_standardization',

            '--validation_set_split_ratio', '0.05',
            '--validate_every_n_epochs', '5',
            '--prelogits_norm_loss_factor', '5e-4',

            
            '--optimizer', 'ADAM', 
            '--learning_rate_schedule_file', 'data/learning_rate_schedule_classifier_casia.txt',
            '--learning_rate', '-1',
            '--weight_decay', '5e-4',

            '--lfw_dir', '/home/ydwu/datasets/white-lfw',
            '--lfw_pairs','/home/ydwu/work/facenet/data/pairs.txt',
            '--lfw_nrof_folds', '2',
            '--lfw_distance_metric', '1',
            '--lfw_use_flipped_images',
            '--lfw_subtract_mean',

            '--pretrained_model', '/home/ydwu/work/facenet/origine-models/20170512-110547/model-20170512-110547.ckpt-250000',
            '--batch_size','30',
            '--epoch_size', '900',
            '--max_nrof_epochs', '150']
            

    print(argv)
    print("-----src.facenet_train_sun--------------")
    args = facenet_train.parse_arguments(argv)
    facenet_train.main(args)


    


            
