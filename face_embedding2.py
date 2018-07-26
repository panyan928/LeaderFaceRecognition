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

#1

def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

class FaceModel:
  def __init__(self, args):
    self.args = args
    model = edict()

    self.threshold = args.threshold
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
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.cpu(), num_worker= 1, accurate_landmark = False, threshold=[0.5,0.5,0.7])
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
    cv2.imwrite('temp/{}.jpg'.format(self.img_num), nimg)
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
    cv2.imwrite('align.jpg',nimg)
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

