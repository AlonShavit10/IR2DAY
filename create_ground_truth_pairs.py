import cv2
import numpy as np
import os
#ir_pic='C:/ir to day/data/evaluation/input/all_pics/small/ir'
#vis_pic='C:/ir to day/data/evaluation/input/all_pics/small/vis'
both_domains_dir='C:/ir to day/data/evaluation/input/night_3/vis_small'
output_dir='C:/ir to day/data/evaluation/input/night_3/both'


img0=2;
img1=2;

count=0
for filename in os.listdir(both_domains_dir):
    if filename.endswith(".jpg"):
        count=count+1
        print(count)
        name=filename
        if count % 2 != 0: # the first file
            img0 = cv2.imread(both_domains_dir+"/"+name)
        else: # the second dile
            img1 = cv2.imread(both_domains_dir+"/"+name)
            both=np.concatenate((img0,img1),axis=1)
            cv2.imwrite(output_dir+"/"+name,both)
            
# get both from unite
#ir_only=both[0:128, 0:128]
#cv2.imshow('one', img1)          
## getIR
#vis_only=both[0:128, 128:256]
#cv2.imshow('two', vis_only)          

            
## get true VIS

        

#img_ir = cv2.imread(ir_pic)
#img_vis = cv2.imread(vis_pic)
#
#cv2.imshow('img1', img0)  
#cv2.imshow('img2', img_vis)  
#
#both=np.concatenate((img_ir,img_vis),axis=1)
#cv2.imshow('both', both)  

