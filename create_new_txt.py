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
import caffe, os, sys, cv2
import mxnet as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py-R-FCN/lib'))
import fast_rcnn
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import argparse
import linecache
import glob
import utils2 as utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'common'))
import face_preprocess
import face_embedding

CLASSES = ('__background__',
           'face')

leader_list = ['DXP', 'JZM', 'MZD', 'XJP', 'PLY']


save_path = './txts_new/'
fp_path = './detect_clean/'

GT = dict()
RT = dict()

def less_average(score):
  num = len(score)
  sum_score = sum(score)
  ave_num = sum_score/num
  # less_ave = [i for i in score if i<ave_num]
  return ave_num

def calculateIoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2]
    gy2 = groundTruthBound[3]

    carea = (cx2 - cx1) * (cy2 - cy1) 
    garea = (gx2 - gx1) * (gy2 - gy1) 

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h 

    iou = area / float(carea + garea - area)

    return iou

def demo(net, leader, label):
    """Detect object classes in an image using pre-computed object proposals."""
    #print image_path
    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    img_name = label.split(';')[0]
    image_path = 'raw/{}-craw/{}.jpg'.format(leader, img_name)
    ### txt path

    gt = utils.get_bbox('txts/' + leader + '/', label, default_leader = leader)
    if gt is None:
    	return
    # x,y,w,h, id, person, pose
    bbs = gt[0]
    for bb in bbs:
    	GT[bb[5]] += 1
    bbs_other = gt[1]
    GT['OTHERS'] += len(bbs_other)
    print( label )
    new_txt_file = open(save_path + leader + '/' + img_name + '.txt', 'w') 
    im = cv2.imread(image_path)
    if im is None:
        image_path = image_path[:-4] + '.JPG'
        im = cv2.imread(image_path)

    if im is None:
        print(image_path + ' is None')
        return

    scores, boxes = im_detect(net, im)

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
            bbox.append(dets[i,:4])
            # cv2.rectangle(im, (dets[i,0], dets[i, 1]), (dets[i, 2], dets[i, 3]), (255, 0, 0))
            # cv2.putText(im, str(dets[i, -1]), (int((dets[i,0]+dets[i,2])/2), int((dets[i, 1]+ dets[i, 3])/2)), 0, 1, (255,0,0))

    for i, d in enumerate(bbox):
        pose = -1
        # bbs: x,y,w,h, id, person, pose
        for j in range(len(bbs)):
            bb = bbs[j]
            bound2 = (float(bb[0]), float(bb[1]),
                      float(bb[0]) + float(bb[2]), float(bb[1])+float(bb[3]))
            if calculateIoU(d, bound2)>=0.5:
                RT[bb[5]] += 1
                f_img_path = fp_path + bb[5] + '/fro/' + img_name + '_{}.jpg'.format(i)
                if os.path.exists(f_img_path):
                    pose = 0
                p_img_path = fp_path + bb[5] + '/pro/' + img_name + '_{}.jpg'.format(i)
                if os.path.exists(p_img_path):
                    pose = 1
                bbs[j][6] = pose
                break

        for j in range(len(bbs_other)):
            bb = bbs_other[j]
            bound2 = (float(bb[0]), float(bb[1]),
                      float(bb[0]) + float(bb[2]), float(bb[1])+float(bb[3]))
            if calculateIoU(d, bound2)>=0.5:
                RT['OTHERS'] += 1
                f_img_path = fp_path + 'OTHERS/fro/' + img_name + '_{}.jpg'.format(i)
                if os.path.exists(f_img_path):
                    pose = 0
                p_img_path = fp_path + 'OTHERS/pro/' + img_name + '_{}.jpg'.format(i)
                if os.path.exists(p_img_path):
                    pose = 1
                bbs_other[j][6] = pose
                break

    for gt in (bbs + bbs_other):
        line = '{} {} {} {} {} {} {}\n'.format(gt[4], gt[0], gt[1], gt[2], gt[3], gt[5], gt[6])
        new_txt_file.write(line)
        ## face that has groundtruth info
        # if len(gt) != 0:
        #     RT[positive] += 1
        #     f_img_path = fp_path + positive + '/fro/' + img_name + '_{}.jpg'.format(i)
        #     if os.path.exists(f_img_path):
        #         pose = 0
        #     p_img_path = fp_path + positive + '/pro/' + img_name + '_{}.jpg'.format(i)
        #     if os.path.exists(p_img_path):
        #         pose = 1
        #     ## pose==-1 if don't exist in detect_clean diretory
        # 	line = '{} {} {} {} {} {} {}\n'.format(gt[4], gt[0], gt[1], gt[2], gt[3], gt[5], pose)
        #     new_txt_file.write(line)
        ## no groundtruth info



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
    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    #im_names = ['pic/1.png','pic/2.jpg','pic/3.jpg']
    GT['OTHERS'] = 0
    RT['OTHERS'] = 0
    for leader in leader_list:
        GT[leader] =0
        RT[leader] =0

    for leader in leader_list:
        if not os.path.exists(save_path + leader):
            os.mkdir(save_path + leader)
        label_file_path = 'result2/{}/{}.txt'.format(leader,leader)
        for label in open(label_file_path):
            if len(label) == 10:
                label = label[:-1]
            # print(name)
            demo(net, leader, label)
    ##print detect result
    print('Groundtruth:')
    print(GT)
    print('Detect Result:')
    print(RT)
