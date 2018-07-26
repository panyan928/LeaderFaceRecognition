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
import caffe, os, sys, cv2, multiprocessing
import face_align
import mxnet as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py-R-FCN/lib'))
# import _init_paths
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
import tensorflow as tf
from nets import nets_factory
slim = tf.contrib.slim
save_root = 'detect_align_cls_all/'

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

def demo(net, image_path, label):
    """Detect object classes in an image using pre-computed object proposals."""
    #print image_path
    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(image_path)
    if im is None:
        image_path = image_path[:-4] + '.JPG'
        im = cv2.imread(image_path)
    leader = image_path.split('/')[1][:3]
    img_name = image_path.split('/')[2][:-4]
    bbs_path = 'txts/' + leader + '/'
    if im is None:
    	print(image_path + ' is None')
    	return
    # Detect all object classes and regress object bounds
    # if os.path.ifexists(save_root + image_name + '*.jpg'):
    # 	return
    bbs, bbs_other = utils.get_bbox(bbs_path, label, default_leader=leader)
    if (len(bbs)+len(bbs_other)) == 0:
        print(image_path + ' have no gt')
        print(label)
        return
    scores, boxes = im_detect(net, im)
    CONF_THRESH = 0.9
    NMS_THRESH = 0.5
    bbox = []
    for cls_ind, cls in enumerate(['face']):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind : 4*(cls_ind + 1)]
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

    # !!! Visualize GroundTurth, Compare and Compute IOU!!!
    # global iou,Sum
    # if os.path.exists('txts/' + leader + '/' + img_name[:-4] +'.txt'):
    #     bbs, bbs_other = utils.get_bbox('txts/' + leader + '/', label, default_leader = leader)
    #     for bb in bbs:
    #         bound2 = (float(bb[0]), float(bb[1]),
    #                   float(bb[0]) + float(bb[2]), float(bb[1])+float(bb[3]))
    #         iou_temp = 0
    #         for d in bbox:
    #             iou_temp = calculateIoU(d, bound2)
    #             if iou_temp >= 0.5:
    #                 break
    #         print(iou_temp)
    #         if iou_temp < 0.5:
    #             cv2.rectangle(im, (int(bound2[0]), int(bound2[1])), (int(bound2[2]), int(bound2[3])), (0, 255, 0))
    #             cv2.imwrite('RFCN_result/NoDetect/'+ leader + '/' + img_name, im)
    #             continue
    #         iou += iou_temp
    #         Sum += 1
    # print(iou / Sum)


    for i, d in enumerate(bbox):
    	# judge whether leader by label
    	iou = 0
        positive = 'OTHERS'
        for bb in bbs:
            bound2 = (float(bb[0]), float(bb[1]),
                      float(bb[0]) + float(bb[2]), float(bb[1])+float(bb[3]))
            iou = calculateIoU(d, bound2)
            if iou >= 0.5:
                positive = bb[5]
                break  

    	preds = fa.get_landmarks(image_path, d)[-1]

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
        # for j in range(0,5):
        #     cv2.circle(im, (five[j], five[j+5]), 1, (0, 0, 255), 2)
        # cv2.imwrite('result.jpg',im)
        # exit(0)
        five = np.array([five])
        #points = points[1,:].reshape((2,5)).T
        five = five.reshape((2, 5)).T
        # width = d[2]-d[0]
        # height = d[3]- d[1]
        # pad = [width/5, height/5]
        # d = (max(0, d[0]- pad[0]), max(0, d[1]-pad[1]), min(im.shape[1], d[2]+pad[0]), min(im.shape[1], d[3]+ pad[1]))
        nimg = face_preprocess.preprocess(im, d, five, image_size='112,112')

        img_temp_cls = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        # img_temp_cls = cv2.resize(img_temp_cls, (299, 299), cv2.INTER_LINEAR)
        img_temp_cls = np.array(img_temp_cls, dtype = np.uint8)
        img_temp_cls = img_temp_cls.astype('float32')
        img_temp_cls = np.multiply(img_temp_cls, 1.0/255.0)
        img_temp_cls = np.subtract(img_temp_cls, 0.5)
        img_temp_cls = np.multiply(img_temp_cls, 2.0)
        img_temp_cls = np.expand_dims(img_temp_cls, 0)
        predictions = sess.run(logits, feed_dict = {tensor_input : img_temp_cls}).indices
        pose = predictions[0]

        if positive=='OTHERS':
            leader = 'OTHERS'
        if positive != leader:
            leader = positive
        if pose == 0:
            save_path = save_root + leader + "/fro/" + img_name + '_{}.jpg'.format(i)
        elif pose == 1:
            save_path = save_root + leader + "/pro/" + img_name + '_{}.jpg'.format(i)
        last_index = save_path.rfind('/')
        if not os.path.exists(save_path[:last_index]):
            print( 'save path \"' + save_path +'\" error')
            exit(0)
        cv2.imwrite(save_path, nimg)

if __name__ == '__main__':
      # Use RPN for proposals
   
    det_prototxt = '/home/disk3/py/py-R-FCN/models/pascal_voc/ResNet-101/rfcn_end2end/test_agonistic_face.prototxt'
    det_caffemodel = '/home/disk3/py/py-R-FCN/data/rfcn_models/resnet101_rfcn_ohem_iter_40000.caffemodel'

    CPU_MODE = False ## set to Ture , occur error " segmention error 11" !!! it cost me  one whole day !
    if CPU_MODE:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        cfg.GPU_ID = 0
    cfg.TEST.HAS_RPN = True
    net = caffe.Net(det_prototxt, det_caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(det_caffemodel)
    fa = face_align.FaceAlignment(face_align.LandmarksType._2D, enable_cuda=False)


    ## classification model

    cls_checkpoint_path = './cls_checkpoint/'
    cls_model_name = 'inception_v3'
    cls_image_size = 112

    network_fn = nets_factory.get_network_fn(
        cls_model_name,
        num_classes=2,
        is_training=False)

    if tf.gfile.IsDirectory(cls_checkpoint_path):
        cls_checkpoint_path = tf.train.latest_checkpoint(cls_checkpoint_path)
        print("Classification Model path:" + cls_checkpoint_path)

    tensor_input = tf.placeholder(tf.float32, [1, cls_image_size, cls_image_size, 3])
    logits, _ = network_fn(tensor_input)
    logits = tf.nn.top_k(logits, 1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.device_count = {'GPU': 0}
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    for leader in ['DXP', 'JZM', 'MZD', 'XJP', 'PLY', 'OTHERS']:
        if not os.path.exists(save_root + leader):
            os.mkdir(save_root + leader)
            os.mkdir(save_root + leader + '/fro')
            os.mkdir(save_root + leader + '/pro')
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, cls_checkpoint_path)
        # Warmup on a dummy image
        #im_names = ['pic/1.png','pic/2.jpg','pic/3.jpg']
        for leader in ['JZM', 'MZD', 'XJP', 'PLY']:
            if not os.path.exists(save_root):
                os.mkdir(save_root)
            label_file_path = 'result2/{}/{}.txt'.format(leader,leader)
            for i, label in enumerate(open(label_file_path)):
                # if int(label.split(';')[0][-4:]) < 1136:
                #     continue
                name = 'raw/{}-craw/{}.jpg'.format(leader, label.split(';')[0])
                print(name)
                demo(net, name, label)

            
 #    im_names = glob.glob('raw/MZD-craw/*.jpg')
 #    im_names += glob.glob('raw/MZD-craw/*.JPG')
 #    im_names_result = glob.glob(save_root + '*.jpg')
 #    #im_names = linecache.getlines(imglist)
 #    flag = True
 #    for im_name in im_names:
 #        print 'Demo for {}'.format(im_name)
 #        # if os.path.exists(save_root + os.path.basename(im_name)[:-4] + '*.jpg'):
 #        # 	print('continue')
 #        # 	continue
	# count += 1
 #        demo(net, im_name)
	# print(count)
 #    print(count)
