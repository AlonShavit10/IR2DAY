import cv2
import os
aim_down1=128
aim_down2=128
skip_files=10

rootdir_vis ='C:/ir to day/data/evaluation/input/night_3/vis'
output_dir_vis='C:/ir to day/data/evaluation/input/night_3/vis_small'
rootdir_ir ='C:/ir to day/data/evaluation/input/night_3/ir'
output_dir_ir='C:/ir to day/data/evaluation/input/night_3/ir_small'
########


counter=0
for subdir, dirs, files in os.walk(rootdir_vis):
    for file in files:
        counter+=1
        if counter%skip_files!=0:
            continue
        
        if file.endswith(".jpg"):
        
       # print (file)
            img = cv2.imread(subdir+"/"+file)
            img = img[0:512, 64:576]
            img1 = cv2.resize(img, (128, aim_down1)) 
            cv2.imwrite(output_dir_vis+"/"+str(counter)+".jpg",img1)
            if counter%20==0:
                print(str(counter))
                
import cv2
import os
aim_down1=128
aim_down2=128
########
counter=0
for subdir, dirs, files in os.walk(rootdir_ir):
    for file in files:
        counter+=1
        if counter%skip_files!=0:
            continue
        
        if file.endswith(".jpg"):
        
       # print (file)
            img = cv2.imread(subdir+"/"+file)
            img = img[0:512, 64:576]
            img1 = cv2.resize(img, (128, aim_down1)) 
            cv2.imwrite(output_dir_ir+"/"+str(counter)+".jpg",img1)
            if counter%20==0:
                print(str(counter))