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
import shutil
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

image_path = '/home/disk3/py/fro-pro/cfp-dataset/Data/Images/'
#image_path = '/ /disk3/py/fro-pro/cfp-dataset/Data/code/'

save_path = '/home/disk3/py/incubator-mxnet/tools/vector_data/'
#save_path = '/home/disk3/py/fro-pro/cfp-dataset/Data/code/save/'

dirs = os.listdir(image_path)

#num = 0
p_num = 0
f_num = 0

for subdir in dirs:
	#print('id:'+subdir)
	os.mkdir(save_path + subdir)
	profile_path = image_path + subdir + '/' + 'profile/'

	frontal_path = image_path + subdir + '/' + 'frontal/'

	os.mkdir(save_path + subdir + '/' + 'profile/')
	os.mkdir(save_path + subdir + '/' + 'frontal/')

	p_image_lists = glob.glob(profile_path +'/*.jpg')
	f_image_lists = glob.glob(frontal_path +'/*.jpg')


	for img_list in p_image_lists:
		print(img_list)
		img_name = os.path.basename(img_list)

		img = cv2.imread(img_list)
		size = img.shape
		print(size)
		bb = [1,1,size[1]-1,size[0]-1]
		preds = fa.get_landmarks(img, bb)[-1]
		#print(img_landmark)

		'''
		five = makeKeyPoints(img_landmark)
		print(five)
		five = array([five])
		five = five.reshape((2, 5)).T
		'''

		f = model.get_feature_by_landmark(img, bb, preds)

		#np.savetxt(save_path + subdir + '/' + 'profile/' + '1.txt', f[0])
		np.savetxt(save_path + subdir + '/' + 'profile/' + img_name.split('.')[0] + '.txt', f[0])

		#np.savetxt(save_path + subdir + '/' + 'profile/' + img_name.split('.')[0] + '.txt', f[0])


		p_num +=1

	for img_list in f_image_lists:
		print(img_list)
		img_name = os.path.basename(img_list)

		img = cv2.imread(img_list)
		size = img.shape
		print(size)
		bb = [1,1,size[1]-1,size[0]-1]
		preds = fa.get_landmarks(img, bb)[-1]
		#print(img_landmark)

		'''
		five = makeKeyPoints(img_landmark)
		print(five)
		five = array([five])
		five = five.reshape((2, 5)).T
		'''

		f = model.get_feature_by_landmark(img, bb, preds)

		np.savetxt(save_path + subdir + '/' + 'frontal/' + img_name.split('.')[0] + '.txt', f[0])


		f_num +=1

print("p_num: %d  f_num: %d"%(p_num,f_num))

print(p_num)
print(f_num)

	#num +=1

#print(num)

