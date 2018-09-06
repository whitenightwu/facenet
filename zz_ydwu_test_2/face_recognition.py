"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

from scipy import misc
import sys
import os
sys.path.append("/home/ydwu/work/facenet/src")
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep


def main(args):
    
    dataset = facenet.get_dataset(args.dataset_dir)
    image_list, label_list = facenet.get_image_paths_and_labels(dataset)
    class_names = [cls.name for cls in dataset]
    
    images = load_and_align_data(image_list, args.image_size, args.margin, args.gpu_memory_fraction, args.detect_multiple_faces)
    print("len(images) = ", len(images))

    
    with tf.Graph().as_default():
        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            
            nrof_images = len(images)
            print("nrof_images = ", nrof_images)
            
            print('Images:')
            for i in range(len(image_list)):
                print('%1d: %s' % (i, image_list[i]))
            print('')

            print("============================")
            print("============================")
            print("============================")

            print("emb.shape = ", emb.shape)
            #print("emb = ", emb)


            print("============================")
            print("============================")
            print("============================")
            
            f = h5py.File(args.dataset_file, 'r')
            # for key in f.keys():
                #print(f[key].name)
                #     #print(f[key].shape)
                #print(f[key].value)


            print("============================")
            print("============================")
            print("============================")

                
            count = 0
            magin = 0.6
            for i in range(nrof_images):
                dist_list = []
                for dataset_embedding in f['embedding']:
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], dataset_embedding[:]))))
                    # print("dist = ", dist)

                    if(dist == 0.0):
                        dist_list.append('NAN')
                        print("Warning!!!")
                    else:
                        dist_list.append(dist)

                print("dist_list = ", dist_list)
                min_dist = min(dist_list)
                print("min_dist = ", min_dist)

                
                # print(dist_list.index(min_dist))
                
                if(min_dist <= magin):
                    count = count + 1
                    print("test_img_dir : ", image_list[i])

                    print("dataset_img_dir : ", f['class_names'][dist_list.index(min_dist)])
                    print("dataset_img_label : ", f['image_list'][dist_list.index(min_dist)])

                    print("True")
                else:
                    print("False")
                print("============================")

                # xx = min_dist.index(min(min_dist))
                # print("i = ", i, " VS ", "min_dist = ", xx)
                # print("i = ", label_list[i], " VS ", "min_dist = ", label_list[xx])
                
                # if label_list[i] == label_list[xx]:

            #     if ((label_list[i] == label_list[xx]) and (min_dist[xx] <= magin)) or ((label_list[i] != label_list[xx]) and (min_dist[xx] > magin)):
            #         print("True")
            #         count = count + 1
            #     else:
            #         print("False")
            #         # print("xxx = ", xxx)
            print("ACC = ", count, " / ", nrof_images)

            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction, detect_multiple_faces):

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

        ###################################
        nrof_faces = bounding_boxes.shape[0]
        print("nrof_faces = ", nrof_faces)
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                if detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                    else:
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                        det_arr.append(det[index,:])
                else:
                    det_arr.append(np.squeeze(det))
                    
                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-margin/2, 0)
                    bb[1] = np.maximum(det[1]-margin/2, 0)
                    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    prewhitened = facenet.prewhiten(scaled)
                    img_list.append(prewhitened)
                    output_filename_n = "{}_{}".format(image[:-4], str(i)+".jpg")
                    print(output_filename_n)
                    misc.imsave(output_filename_n, scaled)
        ###################################



              
        # det = np.squeeze(bounding_boxes[0,0:4])
        # bb = np.zeros(4, dtype=np.int32)
        # bb[0] = np.maximum(det[0]-margin/2, 0)
        # bb[1] = np.maximum(det[1]-margin/2, 0)
        # bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        # bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        # cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        # aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        # prewhitened = facenet.prewhiten(aligned)
        # img_list.append(prewhitened)
        # print(img_list)
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--dataset_dir', type=str,
        help='Path to the directory containing aligned dataset.')
    parser.add_argument('--dataset_file', type=str, help='save embedding')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
