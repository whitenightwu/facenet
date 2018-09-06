"""Performs face alignment and calculates L2 distance between the embeddings of images."""
#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : creat_embedding_dataset.py
## Authors    : ydwu@ydwu-OptiPlex-5050
## Create Time: 2018-08-28:14:40:40
## Description:
## 
##

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append("/home/ydwu/work/facenet/src")

from scipy import misc
import tensorflow as tf
import numpy as np
import os
import copy
import argparse
import facenet
import align.detect_face
import h5py
import math
from six import iteritems
from tensorflow.python.ops import data_flow_ops


def main(args):
    dataset = facenet.get_dataset(args.dataset_dir)
    image_list, label_list = facenet.get_image_paths_and_labels(dataset)
    class_names = [cls.name for cls in dataset]
    #print("============================")
    # print("image_list", image_list)
    # print("label_list", label_list)
    #print("class_names", class_names)
    nrof_images = len(image_list)
    
    images = load_and_align_data(image_list, args.image_size, args.margin, args.gpu_memory_fraction)
    print("len(images) = ", len(images))
    
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            # facenet.load_model(args.model)
            facenet.load_model("/home/ydwu/work/facenet/origine-models/facenet_bake")
            
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


            ##########################
            ##########################
            ##########################
            

            labels_batch = tf.get_default_graph().get_tensor_by_name("label_batch:0")
            batch_size_placeholder = tf.get_default_graph().get_tensor_by_name("batch_size:0")
            learning_rate_placeholder = tf.get_default_graph().get_tensor_by_name("learning_rate:0")
            image_paths_placeholder = tf.get_default_graph().get_tensor_by_name("image_paths:0")
            labels_placeholder = tf.get_default_graph().get_tensor_by_name("labels:0")

            labels_array = np.reshape(np.arange(nrof_images),(-1,1))
            image_paths_array = np.reshape(np.expand_dims(np.array(image_list),1), (-1,1))

            print(len(image_paths_array))
            print(image_paths_array.shape)
            print(len(labels_array))
            print(batch_size_placeholder)
            print(labels_batch)

            print(image_paths_placeholder)
            print(labels_placeholder)
            
            # input_queue = data_flow_ops.FIFOQueue(capacity=100000,
            #                                       dtypes=[tf.string, tf.int64],
            #                                       shapes=[(3,), (3,)],
            #                                       shared_name=None, name=None)
            # enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

            # sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

            sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:False})
            sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:False})

            
            nrof_batches = int(np.ceil(nrof_images / 3))
            for i in range(nrof_batches):
                aa_begin = i*3
                aa_end = i*3 +3
                batch_size = min(nrof_images-i*3, 3)

                aa_image_paths_array = image_paths_array[aa_begin:aa_end]
                aa_labels_array = labels_array[aa_begin:aa_end]
                print(aa_image_paths_array)
                print(aa_labels_array)
                print("=================================================")                
                
                emb = sess.run(embeddings, feed_dict={images_placeholder: aa_image_paths_array, phase_train_placeholder:False})

                # emb = sess.run(embeddings, feed_dict={batch_size_placeholder: batch_size,phase_train_placeholder:False})
                # emb, lab = sess.run([embeddings,labels_batch], feed_dict={image_paths_placeholder: aa_image_paths_array, labels_placeholder: aa_labels_array, batch_size_placeholder: batch_size,phase_train_placeholder:False, learning_rate_placeholder:0.0})
                # emb = sess.run(embeddings, feed_dict={image_paths_placeholder: image_paths_array, batch_size_placeholder: batch_size,phase_train_placeholder:False})
                
                
                #print(emb)
                print(len(emb))
                print(emb.shape)

                print("=================================================")
                mdict = {'class_names':class_names, 'image_list':image_list, 'label_list':lab, 'embedding':emb }
                with h5py.File(args.save_dir, 'wa') as f:
                    #print(mdict)
                    for key, value in iteritems(mdict):
                        #print("============================")
                        #print("value = ", value)
                        #print("key = ", key)
                        f.create_dataset(key, data=value)


    ###
    fff = h5py.File(args.save_dir, 'r')
    for key in fff.keys():
        print(fff[key].name)
        print(fff[key].shape)
        # print(f[key].value)

                        
            ##########################
            ##########################
            ##########################

            # # Run forward pass to calculate embeddings
            # feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            # emb = sess.run(embeddings, feed_dict=feed_dict)

            # #print(emb)
            # print(len(emb))
            # print(emb.shape)

            # #print("=================================================")
            # #print("=================================================")
            # #print("=================================================")
            # mdict = {'class_names':class_names, 'image_list':image_list, 'label_list':label_list, 'embedding':emb }
            # with h5py.File(args.save_dir, 'w') as f:
            #     #print(mdict)
            #     for key, value in iteritems(mdict):
            #         #print("============================")
            #         #print("value = ", value)
            #         #print("key = ", key)
            #         f.create_dataset(key, data=value)



            
            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--dataset_dir', type=str,
        help='Path to the directory containing aligned dataset.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--save_dir', type=str, help='save embedding')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
