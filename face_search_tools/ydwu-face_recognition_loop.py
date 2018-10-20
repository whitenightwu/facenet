#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : zz-white-face_recognition.py
## Authors    : ydwu@ydwu-OptiPlex-5050
## Create Time: 2018-08-29:14:24:00
## Description:
## 
##
import os

#################################

os.system("python ./zz-ydwu-face_recognition.py")

print("#################################")
os.system("sed -i \"s/face_emb_0',/face_emb_1',/g\" ./zz-ydwu-face_recognition.py")
os.system("python zz-ydwu-face_recognition.py")


print("#################################")
os.system("sed -i \"s/face_emb_1',/face_emb_2',/g\" ./zz-ydwu-face_recognition.py")
os.system("python zz-ydwu-face_recognition.py")

print("#################################")
os.system("sed -i \"s/face_emb_2',/face_emb_3',/g\" ./zz-ydwu-face_recognition.py")
os.system("python zz-ydwu-face_recognition.py")

print("#################################")
os.system("sed -i \"s/face_emb_3',/face_emb_4',/g\" ./zz-ydwu-face_recognition.py")
os.system("python zz-ydwu-face_recognition.py")

print("#################################")
os.system("sed -i \"s/face_emb_4',/face_emb_5',/g\" ./zz-ydwu-face_recognition.py")
os.system("python zz-ydwu-face_recognition.py")

print("#################################")
os.system("sed -i \"s/face_emb_5',/face_emb_6',/g\" ./zz-ydwu-face_recognition.py")
os.system("python zz-ydwu-face_recognition.py")

print("#################################")
os.system("sed -i \"s/face_emb_6',/face_emb_7',/g\" ./zz-ydwu-face_recognition.py")
os.system("python zz-ydwu-face_recognition.py")

print("#################################")
os.system("sed -i \"s/face_emb_7',/face_emb_0',/g\" ./zz-ydwu-face_recognition.py")


# print("#################################")
# os.system("sed -i \"s/face_emb_7',/face_emb_8',/g\" ./zz-ydwu-face_recognition.py")
# os.system("python zz-ydwu-face_recognition.py")

# print("#################################")
# os.system("sed -i \"s/face_emb_8',/face_emb_9',/g\" ./zz-ydwu-face_recognition.py")
# os.system("python zz-ydwu-face_recognition.py")

# print("#################################")
# os.system("sed -i \"s/face_emb_9',/face_emb_10',/g\" ./zz-ydwu-face_recognition.py")
# os.system("python zz-ydwu-face_recognition.py")

# print("#################################")
# os.system("sed -i \"s/face_emb_10',/face_emb_11',/g\" ./zz-ydwu-face_recognition.py")
# os.system("python zz-ydwu-face_recognition.py")

# print("#################################")
# os.system("sed -i \"s/face_emb_11',/face_emb_12',/g\" ./zz-ydwu-face_recognition.py")
# os.system("python zz-ydwu-face_recognition.py")

# print("#################################")
# os.system("sed -i \"s/face_emb_12',/face_emb_13',/g\" ./zz-ydwu-face_recognition.py")
# os.system("python zz-ydwu-face_recognition.py")

# print("#################################")
# os.system("sed -i \"s/face_emb_13',/face_emb_14',/g\" ./zz-ydwu-face_recognition.py")
# os.system("python zz-ydwu-face_recognition.py")

# print("#################################")
# os.system("sed -i \"s/face_emb_14',/face_emb_0',/g\" ./zz-ydwu-face_recognition.py")

