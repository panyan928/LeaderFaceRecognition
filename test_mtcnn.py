from __future__ import division
#import face_align 
#import face_preprocess
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import scipy.io as sio
from numpy import*
import glob
#import utils 
import random
import face_embedding2
import argparse 


def make_score(num):
  score = [random.randint(0,100) for i in range(num)]
  return score
 
def less_average(score):
  num = len(score)
  sum_score = sum(score)
  ave_num = sum_score/num
  # less_ave = [i for i in score if i<ave_num]
  return ave_num

class Config:
    leaders_sml = ['DXP','JZM', 'MZD', 'PLY', 'XJP', 'OTHERS']#,
    save_root = 'align_result'
    label_path_root = '/home/disk3/py/face_data/result2'
    bbs_path_root = '/home/disk3/py/face_data/txts'
    img_path_root = '/home/disk3/py/face_data/raw'

#fa = face_align.FaceAlignment(face_align.LandmarksType._2D, enable_cuda=True)
# paths = glob.glob("~/face_data/raw/DXP-craw/*")
cfg = Config()



parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./model-r50-am-lfw/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.5, type=float, help='ver dist threshold')
args = parser.parse_args()


path = '1.jpg'

img = cv2.imread(path)
'''
if img is None:
    path = img_path + img_name + '.JPG'
    img = cv2.imread(path)
    if img is None:
        print("Can't find image")
        continue
'''

input = io.imread(path)
input2 = io.imread(path)
height, width, c = img.shape


model = face_embedding2.FaceModel(args)
boxes, points = model.detect(input)
#ret = fe.detect_face_limited(input, det_type = self.args.det)
#bbox, points = result
#bbox = bbox[0,0:4]

print(boxes)
print(points)



bb = [2,1,150,116]


#point = [42, 74, 84, 79, 60, 101, 42, 119, 66, 115]
#point = [42, 84, 60, 42, 66, 74, 79, 101, 119, 115]
#point=[21,45,24,45,13,55,19,68,22,67]
point = [21, 24, 13, 19, 22, 45, 45, 55, 68, 67]
point = array([point])
print(point)
f = model.get_feature(input, bb, point)


#for d, point in zip(boxes,points):
#  f = model.get_feature(input, d, point)







'''
bbs, bbs_other = utils.get_bbox(bbs_path, label, default_leader=leader)
for bb in (bbs + bbs_other):
    # print(bb)
    b_width = float(bb[2])
    b_height = float(bb[3])

    bb[0]= float(bb[0])
    bb[1]=float(bb[1])
    bb[2]=float(bb[2])+float(bb[0])
    bb[3]=float(bb[3])+float(bb[1])
    # img = img[bb[1]:bb[3],bb[0]:bb[2]]
    print(bb[:4])
'''

#bb = [5,7,93,92]
#preds = fa.get_landmarks(input, bb[:4])[-1]
#preds = fa.get_landmarks(input)[-1]

# print(preds)
'''
lefteye_x =int(less_average([preds[36,0],preds[37,0],preds[38,0],preds[39,0],preds[40,0],preds[41,0]]))
lefteye_y =int(less_average([preds[36,1],preds[37,1],preds[38,1],preds[39,1],preds[40,1],preds[41,1]]))

righteye_x =int(less_average([preds[42,0],preds[43,0],preds[44,0],preds[45,0],preds[46,0],preds[47,0]]))
righteye_y =int(less_average([preds[42,1],preds[43,1],preds[44,1],preds[45,1],preds[46,1],preds[47,1]]))

nose_x =int(less_average([preds[27,0],preds[28,0],preds[29,0],preds[30,0],preds[32,0],preds[33,0],preds[34,0]]))
nose_y =int(less_average([preds[27,1],preds[28,1],preds[29,1],preds[30,1],preds[32,1],preds[33,1],preds[34,1]]))

leftmouth_x =int(less_average([preds[48,0],preds[49,0],preds[60,0],preds[59,0]]))
leftmouth_y =int(less_average([preds[48,1],preds[49,1],preds[60,1],preds[59,1]]))

rightmouth_x =int(less_average([preds[54,0],preds[53,0],preds[64,0],preds[55,0]]))
rightmouth_y =int(less_average([preds[54,1],preds[53,1],preds[64,1],preds[55,1]]))
'''

'''
cv2.rectangle(img, (int(bb[0]),int(bb[1])), (int(bb[2]),int(bb[3])), (0, 0, 255), 2)
port = b_width / abs(righteye_x-lefteye_x)
if port > 3:
    cv2.putText(img, "{:.1f} 0".format(port), (int(bb[0]),int(bb[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
else:
    cv2.putText(img, "{:.1f} 0".format(port), (int(bb[0]),int(bb[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
# cv2.imwrite(save_root + img_name +'.jpg',img)
'''

five = []
five.append(lefteye_x)
five.append(lefteye_y)
five.append(righteye_x)
five.append(righteye_y)
five.append(nose_x)
five.append(nose_y)
five.append(leftmouth_x)
five.append(leftmouth_y)
five.append(rightmouth_x)
five.append(rightmouth_y)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(input)
for i in range(0,10,2):
    ax.plot(five[i],five[i+1], marker ='o', markersize=6, linestyle='-',color='w',lw=2)
    ax.text(five[i], five[i+1], str(i), fontsize=5, color='r')
    print(str(five[i])+' '+str(five[i+1]))
# for i in range(0,68):
#     ax.plot(preds[i, 0],preds[i, 1], marker ='o', markersize=6, linestyle='-',color='w',lw=2)
#     ax.text(preds[i,0], preds[i,1], str(i), fontsize=5, color='r')
ax.axis('off')
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()
# plt.savefig()

# print(points)
#	points = mat(points)
# print(five)

for i in range(0,10,2):
    cv2.circle(img, (five[i], five[i+1]), 1, (0, 0, 255), 2)
#cv2.imshow('opencv',img)
cv2.imwrite('2.jpg',img)
#points = points[1,:].reshape((2,5)).T

five = array([five])
five = five.reshape((2, 5)).T
#p = [[]]
#p[0]= points[1:]
#points = p.reshape((2,5)).T

nimg = face_preprocess.preprocess(input2, bb[:4], five, image_size='112,112')
# cv2.imwrite(save_root + img_name + "_" + bb[4] + '.jpg', nimg)


cv2.imwrite('3.jpg',nimg)
#cv2.imwrite(save_root + img_name +'.jpg', img)
# cv2.waitKey(0)
# cv2.imwrite('temp/{}_{}.jpg'.format(imname, i), nimg)
