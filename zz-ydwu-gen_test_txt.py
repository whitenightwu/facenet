import os
# path = r'C:\Users\zhizh\Desktop\face_near_infrared_test _v3'

# #########################
# path = '/home/ydwu/datasets/face_near_infrared_test _v5/INF'
# path_RGB = '/home/ydwu/datasets/face_near_infrared_test _v5/RGB'

# count = 1
# for person in os.listdir(path):
#     for each_img in os.listdir(path+'/'+person):
#         os.rename(path+'/'+person+'/'+each_img,path+'/'+person+'/'+str(count)+'.png')
#         os.rename(path_RGB+'/'+person+'/'+each_img,path_RGB+'/'+person+'/'+str(count)+'.png')
#         count+=1
#     count = 1




#########################
path = '/home/ydwu/datasets/face_near_infrared_test/RGB'

l=[]
for person in os.listdir(path):
    for each_img in os.listdir(path+'/'+person):
        l.append(person+'/'+each_img.strip('.png'))

with open('nir2.txt','a') as f:
    for i in l:
        for j in l:
            if i != j and i.split('/')[0] == j.split('/')[0]:
                f.write(i.split('/')[0]+' '+i.split('/')[1]+' '+j.split('/')[1]+'\n')

            elif i != j and i.split('/')[0] != j.split('/')[0]:
                f.write(i.split('/')[0]+' '+i.split('/')[1]+' '+j.split('/')[0]+' '+j.split('/')[1]+'\n')
