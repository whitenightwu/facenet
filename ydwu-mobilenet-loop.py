#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : ydwu-try.py
## Authors    : ydwu@ydwu-OptiPlex-5050
## Create Time: 2018-08-17:16:40:36
## Description:
## 
##
import os

#################################
print "First set of parameters"

os.system("python ./ydwu-mobilenet.py")

#################################
print "Second set of parameters"

os.system("sed -i \"s/--learning_rate','0.01'/--learning_rate','0.005'/g\" ./ydwu-mobilenet.py")

os.system("python ./ydwu-mobilenet.py")

#################################
print "Thrid set of parameters"

os.system("sed -i \"s/--optimizer', 'RMSPROP'/--optimizer', 'ADAM'/g\" ./ydwu-mobilenet.py")

os.system("python ./ydwu-mobilenet.py")



# #################################
# print "Forth set of parameters"

# os.system("sed -i \"s/--learning_rate','0.5'/--learning_rate','0.1'/g\" ./ydwu-mobilenet.py")

# os.system("python ./ydwu-mobilenet.py")


# #################################

# os.system("sed -i \"s/--learning_rate','0.1'/--learning_rate','1'/g\" ./ydwu-mobilenet.py")
