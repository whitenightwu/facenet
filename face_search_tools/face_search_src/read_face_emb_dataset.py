#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : tmp.py
## Authors    : ydwu@ydwu-OptiPlex-5050
## Create Time: 2018-08-29:12:49:30
## Description:
## 
##
import sys
from scipy import misc
import tensorflow as tf
import numpy as np
import os
import copy
import argparse
import h5py
import math
from six import iteritems


# with h5py.File("/home/ydwu/work/facenet/zz_ydwu_test_2/face_emb_v20", 'r') as f:    
#     for key in f.keys():
#         print(f[key].name)
#         print(f[key].shape)
#         # print(f[key].value)

#     print(f['image_dir'][0])
#     print(f['image_dir'][-1])
# print("====================================")

# with h5py.File("/home/ydwu/work/facenet/zz_ydwu_test_2/face_emb_v21", 'r') as f:    
#     for key in f.keys():
#         print(f[key].name)
#         print(f[key].shape)
#         # print(f[key].value)

#     print(f['image_dir'][0])
#     print(f['image_dir'][-1])
# print("====================================")


##################################
# a_pwd = "/home/ydwu/work/facenet/zz_ydwu_test_2/face_emb_dataset"
a_pwd = "/tmp/ydwu-face"
face_emb_dataset = os.listdir(a_pwd)
print(face_emb_dataset)
for face_emb_name in face_emb_dataset:
    print("====================================")
    print(face_emb_name)
    face_emb = os.path.join(a_pwd + '/' + face_emb_name)
    print(face_emb)
    with h5py.File(face_emb, 'r') as f:    
        for key in f.keys():
            print(f[key].name)
            print(f[key].shape)
            # print(f[key].value)
        print(f['image_dir'][0])
        print(f['image_dir'][-1])
