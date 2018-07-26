from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'common'))
import face_image
import face_preprocess
import numpy

def less_average(score):
  num = len(score)
  sum_score = sum(score)
  ave_num = sum_score/num
  # less_ave = [i for i in score if i<ave_num]
  return ave_num

def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def calEuclideanDistance(vec1,vec2):
    dist = np.sqrt(np.sum(np.square(vec1-vec2)))
    return dist


class FaceModel:
  def __init__(self, args):
    self.args = args
    model = edict()
    self.det_minsize = 50
    self.det_threshold = [0.4,0.6,0.6]
    self.det_factor = 0.9
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.image_size = image_size
    _vec = args.model.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    ctx = mx.gpu(0)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.gpu(0), num_worker= 4, accurate_landmark = False)
    detector_limited = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker= 1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector
    self.detector_limited = detector_limited
    self.img_num = 0

  def get_feature_limited(self, face_img):
    
    result = self.detector_limited.detect_face_limited(face_img,  det_type = self.args.det)
    if result is None:
      return None
    bbox, points = result
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    # cv2.imwrite('temp/{}.jpg'.format(self.img_num), nimg)
    self.img_num += 1
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    #print(nimg.shape)
    embedding = None
    for flipid in [0,1]:
      if flipid==1:
        if self.args.flip==0:
          break
        do_flip(aligned)
      input_blob = np.expand_dims(aligned, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      _embedding = self.model.get_outputs()[0].asnumpy()
      #print(_embedding.shape)
      if embedding is None:
        embedding = _embedding
      else:
        embedding += _embedding
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding


  def get_feature(self, face_img, bbox, points):
    #face_img is bgr image
    # ret = self.detector.detect_face_limited(face_img, det_type = self.args.det)
    # if ret is None:
    #   return None
    # bbox, points = ret
    # if bbox.shape[0]==0:
    #   return None
    bbox = bbox[0:4]
    points = points.reshape((2,5)).T
    #print(bbox)
    #print(points)
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    #print(nimg.shape)
    embedding = None
    for flipid in [0,1]:
      if flipid==1:
        if self.args.flip==0:
          break
        do_flip(aligned)
      input_blob = np.expand_dims(aligned, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      _embedding = self.model.get_outputs()[0].asnumpy()
      #print(_embedding.shape)
      if embedding is None:
        embedding = _embedding
      else:
        embedding += _embedding
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

  def detect(self, face_img):
    results = self.detector.detect_face(face_img)
    if results is not None:

      total_boxes = results[0]
      points = results[1]

      return total_boxes, points
    else:
      return None,None
    
  def get_feature_by_landmark(self, face_img, bb,preds):
  ## choose five landmarks from 67 points
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
    five = []
    five.append(lefteye_x)
    five.append(righteye_x)
    five.append(nose_x)
    five.append(leftmouth_x)
    five.append(rightmouth_x)
    five.append(lefteye_y)
    five.append(righteye_y)            
    five.append(nose_y)            
    five.append(leftmouth_y)            
    five.append(rightmouth_y)
    five = np.array([five])
    five = five.reshape((2, 5)).T
    nimg = face_preprocess.preprocess(face_img, bb, five, image_size='112,112')
    # cv2.imwrite('temp/' + str(self.img_num) +'.jpg', nimg)
    self.img_num += 1
    aligned = np.transpose(nimg, (2,0,1))
    #print(nimg.shape)
    embedding = None
    for flipid in [0,1]:
      if flipid==1:
        if self.args.flip==0:
          break
        do_flip(aligned)
      input_blob = np.expand_dims(aligned, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      _embedding = self.model.get_outputs()[0].asnumpy()
      #print(_embedding.shape)
      if embedding is None:
        embedding = _embedding
      else:
        embedding += _embedding
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding, nimg

  def calEuclideanDistance(vec1,vec2):  
    dist = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))  
    return dist

  def get_fp_by_landmark(self, face_img, bb,preds):
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



    p1 = [lefteye_x,lefteye_y]  
    p2 = [righteye_x,righteye_y]
    p3 = [nose_x,nose_y]  
    p1 = numpy.array(p1)  
    p2 = numpy.array(p2)  
    p3 = numpy.array(p3)

    a = calEuclideanDistance(p1,p2)
    b = calEuclideanDistance(p1,p2)
    c = calEuclideanDistance(p2,p3)

    if ((np.square(a) + np.square(b)) < np.square(c)) or ((np.square(a) + np.square(c)) < np.square(b)):
      print('profile\n')
      return 'p'
    else:
      print('frontal\n')
      return 'f'



