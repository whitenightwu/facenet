import sys
import os
import copy
import tensorflow as tf
import numpy as np
import facenet.src.align.detect_face as facenet_detect_face
from facenet.src import facenet
import imageio
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

_dir = os.path.dirname(os.path.abspath(__file__))

def detect_faces(image_paths, image_size=160, margin=44):
    minsize = 20 # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709 # scale factor
    gpu_memory_fraction = 0.5

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = facenet_detect_face.create_mtcnn(sess, None)

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    result_list = []
    tmp_image_paths = copy.copy(image_paths)
    for image in tmp_image_paths:

        img = imageio.imread(os.path.expanduser(image), pilmode='RGB')
        bounding_boxes, points = facenet_detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            print("can't detect face, remove ", image)
        
        source_img = Image.open(os.path.expanduser(image))
        for i in range(bounding_boxes.shape[0]):
            
            draw = ImageDraw.Draw(source_img)
            draw.rectangle(bounding_boxes[i,0:4].tolist(), outline="lime")
            font_location = bounding_boxes[i,0:2] - np.array([0, 30])
            confidence = "{:.6f}".format(bounding_boxes[i,4] * 100)
            draw.text(font_location, str(confidence) + "%", fill="white", font=ImageFont.truetype("arial", 20))
            for j in range(5):
                point_x = points[j,i]
                point_y = points[j+5,i]
                r = 2
                draw.ellipse((point_x-r, point_y-r, point_x+r, point_y+r), fill="lime")
        
        source_img.save(os.path.splitext(os.path.expanduser(image))[0] + "_result.jpg", "JPEG")
            
    return result_list

if __name__ == '__main__':
    detect_faces(['1.jpg'])
