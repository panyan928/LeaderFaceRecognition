from __future__ import division
import face_align
import cv2
import numpy as np
import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import scipy.io as sio
from numpy import*
import glob
import random
import os
import face_embedding, argparse


parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/disk3/py/face_data/model-r50-am-lfw/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
args = parser.parse_args()
model = face_embedding.FaceModel(args)




fa = face_align.FaceAlignment(face_align.LandmarksType._2D, enable_cuda=True)

img_path1 = '/home/disk3/py/fro-pro/cfp-dataset/Data/Images/001/profile/01.jpg'
img_path2 = '/home/disk3/py/fro-pro/cfp-dataset/Data/Images/001/profile/02.jpg'

img = cv2.imread(img_path1)
size = img.shape
bb = [1,1,size[1]-1,size[0]-1]

img_landmark = fa.get_landmarks(img, bb)[-1]

f1 = model.get_feature_by_landmark(img, bb, img_landmark)

img2 = cv2.imread(img_path2)
size2 = img2.shape
bb2 = [1,1,size[1]-1,size[0]-1]

img_landmark2 = fa.get_landmarks(img2, bb2)[-1]

f2 = model.get_feature_by_landmark(img2, bb2, img_landmark2)

dist = np.sum(np.square(f1[0]-f2[0]))

np.savetxt('f2.txt',f1[0])

print(f1[0].dtype)
print(f1[0].shape)
print(dist)

