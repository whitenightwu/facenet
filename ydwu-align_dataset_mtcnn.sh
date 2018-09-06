

# python align_dataset_mtcnn.py  \
#        --image_size=64 \
#        --gpu_memory_fraction=0.7 \
#        --random_order \
#        --detect_multiple_faces=False\
#        --input_dir=/home/ydwu/datasets/lfw \
#        --output_dir=/home/ydwu/datasets/white-lfw 


##########################################
################### lfw
##########################################
# python src/align/align_dataset_mtcnn.py \
# /home/ydwu/datasets/lfw \
# /home/ydwu/datasets/white-lfw \
# --image_size 182 \
# --margin 44 \
# --gpu_memory_fraction 0.6


##########################################
################### ms1mclean
##########################################
# python src/align/align_dataset_mtcnn.py \
# /media/ydwu/Document/Datasets/ms1mclean \
# /media/ydwu/Document/Datasets/white-ms1mclean \
# --image_size 182 \
# --margin 44


# python src/align/align_dataset_mtcnn.py \
# /media/ydwu/Document/Datasets/ms1mclean-2 \
# /media/ydwu/Document/Datasets/white-ms1mclean-2 \
# --image_size 182 \
# --margin 44


##########################################
##########################################
##########################################
CUDA_VISIBLE_DEVICES="" \
python src/align/align_dataset_mtcnn.py \
/home/ydwu/work/facenet/zz_ydwu_test_2/test_dataset \
/home/ydwu/work/facenet/zz_ydwu_test_2/test_dataset_align-v2 \
--image_size 182 \
--margin 12 \
--detect_multiple_faces True \
--gpu_memory_fraction 0.6

