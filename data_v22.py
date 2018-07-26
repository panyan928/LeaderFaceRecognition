from __future__ import division
import numpy as np
import glob
import os
import face_embedding, argparse
import cv2
import face_align
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/disk3/xmq/leader_data/data_create/fp_model/profile/profile_model,700', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
args = parser.parse_args()
model = face_embedding.FaceModel(args)

fa = face_align.FaceAlignment(face_align.LandmarksType._2D, enable_cuda=True)



#pic_path = '/home/disk3/py/fro-pro/data_clean/'

pic_path = '/home/disk3/xmq/leader_data/data_create/leader_data/'

save_path = '/home/disk3/xmq/leader_data/data_create/data_txt/'

p_num = 0
f_num = 0

for f in ['pro']:
	#print(f)
	pic_path2 = pic_path + f +'/'

	os.mkdir(save_path + f + '/')


	dirs = os.listdir(pic_path2)
	for subdir in dirs:
		print(subdir)

		save_path2 = save_path + f + '/' + subdir + '/'

		if not os.path.exists(save_path2):
			print('path not exists')
			os.mkdir(save_path2)
		else:
			print('path exists')

		pic_path_3 = pic_path2 + subdir + '/'

		pic_lists = glob.glob(pic_path_3 + '/*.jpg')

		for pic_list in pic_lists:
			pic_name = os.path.basename(pic_list)
			print(pic_name)

			img = cv2.imread(pic_list)
			size = img.shape
			print(size)
			bb = [1,1,size[1]-1,size[0]-1]
			preds = fa.get_landmarks(img, bb)[-1]

			v = model.get_feature_by_landmark(img, bb, preds)

			pic_name2 = pic_name.split('.')[0]

			np.savetxt(save_path2 + pic_name2 + '.txt' , v[0])

			print(f)

			if f == 'pro':
				p_num +=1
			elif f == 'fro':
				f_num +=1
			else:
				print('error')

print('p_num: %d'%p_num)
print('f_num: %d'%f_num)




