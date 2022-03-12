

import numpy as np 
import pandas as pd 
import os
import glob
import csv



POSE_PATH_DIR = "/home/negar/Desktop/Pooya/CVPR21Chal-SLR/new/more_new/mmpose/out"
treshold = 0.5

u = 0
with open('./sign_hand_detection.csv', 'w') as f:
    writer = csv.writer(f)

    for path, subdirs, files in os.walk(POSE_PATH_DIR):
        for file in subdirs:
            if file in ['misc_2', 'deafvideo_3', 'deafvideo_2', 'misc_1', 'youtube_1', 'deafvideo_1', 'gallaudet', 'deafvideo_6', 'youtube_2', 'awti', 'aslthat', 'youtube_6', 'deafvideo_5', 'aslized', 'youtube_3', 'deafvideo_4', 'youtube_4', 'youtube_5']:
                    continue
            left_joints = []
            right_joints = []

            for frame in sorted(glob.glob(path+"/"+file+"/*")):
                # print(frame)
                frame_np = np.load(frame,allow_pickle=True)

                if frame_np is None or len(frame_np) ==0:
                    continue

                right_hand = frame_np[0]["keypoints"][112:132]
                left_hand = frame_np[0]["keypoints"][91:112]


                right_hand[:,:2] = right_hand[:,:2] - (sum(right_hand)/len(right_hand))[:2]
                left_hand[:,:2] = left_hand[:,:2] - (sum(left_hand)/len(left_hand))[:2]

                left_joints.append(left_hand)
                right_joints.append(right_hand)

            left_joints = np.array(left_joints)
            right_joints = np.array(right_joints)

            left_conf = np.mean(np.mean(left_joints,axis=0),axis=0)[2]
            right_conf = np.mean(np.mean(right_joints,axis=0),axis=0)[2]

            left_var = sum(sum(abs(left_joints[left_joints.shape[0]//10+1:]-left_joints[left_joints.shape[0]//10:-1])))
            right_var = sum(sum(abs(right_joints[right_joints.shape[0]//10+1:]-right_joints[right_joints.shape[0]//10:-1])))

            if file == "aidan_mack_3173":
                print(left_conf,right_conf)
                print(left_var)
                print(right_var)

            if left_conf<treshold and right_conf <treshold :
                print(left_conf)
                print(right_conf)
                print(path,file,"no hand detected")

            if left_conf<treshold :
                # right hand 
                data = [path,file,"r"]
                writer.writerow(data)
            elif right_conf <treshold :
                # left hand 
                data = [path,file,"l"]
                writer.writerow(data)
            else:
                if left_var[0]+left_var[1] > right_var[0]+right_var[1]:
                    # left hand 
                    data = [path,file,"l"]
                    writer.writerow(data)
                else:
                    # right hand 
                    data = [path,file,"r"]
                    writer.writerow(data)

        # for name in files:
        #     full = os.path.join(path, name)
