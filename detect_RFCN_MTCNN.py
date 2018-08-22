#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import mxnet as mx
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import linecache
import glob

from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'common'))
import face_preprocess
import face_embedding
mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')

detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.gpu(0), num_worker= 1, accurate_landmark = True, threshold=[0.0,0.0,0.2])

CLASSES = ('__background__',
           'face')

count=1


def mtcnn_feature(im, imname, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print('0')
        return
    print(len(inds))
    height, width , _ = im.shape
    for i in inds:
        bbox = dets[i, :4]
        bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        score = dets[i, -1]
        img_temp = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        result = detector.detect_face_limited(img_temp,  det_type = 2)
        pad = 0
        if result is None:
            # zoom in / out the rectangle if can't detect by MTCNN
            # - : min(height, width) / 3
            # + : max(height, width)
            temp1 = min(bbox[3]-bbox[1], bbox[2]-bbox[0]) / 3
            temp2 = max(bbox[3]-bbox[1], bbox[2]-bbox[0]) + 1

            for j in range(1, temp2):
                # assure img_temp in img
                if j == 0: continue
                img_temp = im[max(0, bbox[1] - j):min(bbox[3] + j, height),
                           max(0, bbox[0] - j):min(bbox[2] + j, width)]
                result = detector.detect_face_limited(img_temp,  det_type = 2)
                if result is not None:
                    pad = j
                    break
        scale = 0.5#(10/(bbox[2]-bbox[0]))
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),3)  
        cv2.putText(im,str(score),(bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX,scale,(0,255,0),1)
        print("aaa")
        if result is None: 
            continue
        bbox, points = result
        print(bbox,points)
        bbox = bbox[0,0:4]
        for j in range(5):
            cv2.circle(img_temp, (points[0,j], points[0, j + 5]), 1, (0, 0, 255), 2)
        points = points[0,:].reshape((2,5)).T
        nimg = face_preprocess.preprocess(img_temp, bbox, points, image_size='112,112')
        cv2.imwrite('temp/{}_{}.jpg'.format(imname, i), nimg)

        
        #cv2.imwrite('result/re_'+imname[imname.rfind('/')+1:],im)
    cv2.imwrite('temp/{}'.format(imname),im)
    print(imname + ' saved!')


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    #print image_name
    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(image_name)
    image_name = os.path.basename(image_name)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.9
    NMS_THRESH = 0.5
    bbox = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in inds:
            print(dets[i,0], dets[i,1], dets[i,2]-dets[i,0], dets[i,3]-dets[i, 1])
        # mtcnn_feature(im, image_name, cls, dets, thresh=CONF_THRESH)

if __name__ == '__main__':
      # Use RPN for proposals
   
    prototxt = '/home/disk3/py/py-R-FCN/models/pascal_voc/ResNet-101/rfcn_end2end/test_agonistic_face.prototxt'    
    caffemodel = '/home/disk3/py/py-R-FCN/data/rfcn_models/resnet101_rfcn_ohem_iter_40000.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    CPU_MODE = False
    if CPU_MODE:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        cfg.GPU_ID = 0
    cfg.TEST.HAS_RPN = True
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print('\n\nLoaded network {:s}'.format(caffemodel))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    print('detect dummy image success')
    #im_names = ['pic/1.png','pic/2.jpg','pic/3.jpg']
    im_names = glob.glob('raw/DXP-craw/*.jpg')
    #im_names = linecache.getlines(imglist)
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' )
        im_name = 'raw/XJP-craw/XJP_02515.jpg'
        print ('Demo for {}'.format(im_name))
        demo(net, im_name)
        exit()
        count+=1
    print(count)
