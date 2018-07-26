#!/usr/bin/env python

# -*- coding:utf-8 -*-
from skimage import io
import utils2 as utils
import glob, os
import numpy as np
import cv2
import face_embedding, argparse, shutil
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import caffe

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

class Config:
    leaders_sml = ['DXP','JZM', 'MZD', 'PLY', 'XJP', 'OTHERS']#,
    save_root = './detect_compute'
    label_path_root = './result2'
    bbs_path_root = './txts'
    img_path_root = './raw'
    def BaseFeat(self, leader):
        feature = list()
        bases = glob.glob('raw2/' + leader + '-base/' +leader+'_*.jpg')
        # bases = glob.glob('raw2/' + leader + '-base/*.jpg')
        for n, path in enumerate(bases):
            label = os.path.basename(path)[:-4] + ';0;'
            
            if label[7:9] =='61':
                img = cv2.imread(path)
                img = img[:, img.shape[1]/2:img.shape[1]]
                boxes,points = model.detect(img)
                f = model.get_feature(img, boxes[0], points[0])
                img = img[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]
                cv2.imwrite('DXP_00061_2.jpg', img)
                if f is None: print('DXP_00061.jpg is None')
                feature.append(f)
                continue

            img = cv2.imread(path)
            bbs, bbs_other = utils.get_bbox('txts/' + leader + '/', label, default_leader=leader)
            for i in range(0, int(max(float(bbs[0][2]), float(bbs[0][3])))):
                bbs_temp = (max(0, int(float(bbs[0][0])) - i), 
                            max(0, int(float(bbs[0][1])) - i),
                            min(img.shape[1], int(float(bbs[0][0]) + float(bbs[0][2]) + i)),
                            min(img.shape[0], int(float(bbs[0][1]) + float(bbs[0][3]) + i)))
                img_temp = img[bbs_temp[1]:bbs_temp[3], bbs_temp[0]:bbs_temp[2]]
                # cv2.imshow("crop", img_temp)
                # cv2.waitKey(0)
                f = model.get_feature_limited(img_temp)
                if f is not None:
                    # cv2.imwrite('raw2/' + leader + '-base/' +str(n)+'.jpg', img_temp)
                    break
            if f is None: print(path + ' is None')
            feature.append(f)
        return feature

if __name__ == '__main__':
    # load model
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='./model-r50-am-lfw/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.5, type=float, help='ver dist threshold')
    args = parser.parse_args()

    model = face_embedding.FaceModel(args)

    # face detector model

    #prototxt = os.path.join( args.proto )
   
    prototxt = '/home/disk3/py/py-R-FCN/models/pascal_voc/ResNet-101/rfcn_end2end/test_agonistic_face.prototxt'
     
    #caffemodel = os.path.join( args.demo_net )
    
    caffemodel = '/home/disk3/py/py-R-FCN/data/rfcn_models/resnet101_rfcn_ohem_iter_40000.caffemodel'
    cfg.TEST.HAS_RPN = True
    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print ('\n\nLoaded network {:s}'.format(caffemodel))
    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    # leader feature
    feat_box = dict()
    cfg = Config()
    # boundingboxs numer of all leaders

    output = open('result_txt/result_RFCN_sin_2.txt', 'w')
    output.write('total leaders: {}'.format(len(cfg.leaders_sml[:-1])))
    countDet = 0
    countData = [0, 0]
    no_label = 0
    right = [0, 0]
    FN = 0
    FP = 0
    precision = 0
    recall = 0
    num = 0

    for leader in cfg.leaders_sml[:-1]:
    	save_root = '{}/{}/'.format(cfg.save_root, leader)
        f1 = cfg.BaseFeat(leader)
        feat_box[leader] = f1
        shutil.rmtree(save_root)
        os.mkdir(save_root)
        os.mkdir(save_root + 'no_label')
        os.mkdir(save_root + 'no_detect')
        os.mkdir(save_root + 'falsePositive')
        os.mkdir(save_root + 'falseNegative')

    for leader in ['DXP']:

        output.write(leader)
        # result2/DXP/DXP.txt
        label_file_path = '{0}/{1}/{1}_new.txt'.format(cfg.label_path_root, leader)
        # txts/DXP/
        bbs_path = '{}/{}/'.format(cfg.bbs_path_root, leader)
        # raw/DXP-craw/
        img_path = '{}/{}-craw/'.format(cfg.img_path_root, leader)
        # detect_result/DXP/
        save_root = '{}/{}/'.format(cfg.save_root, leader)

        for label in open(label_file_path):
            if len(label) <= 10:
                continue
            print(label)
            # if int(label[6:9]) != 23: continue
            img_name = label.split(';')[0]
        
            img_cv = cv2.imread(img_path + img_name + '.jpg')
            if img_cv is None:
                img_cv = cv2.imread(img_path+img_name +'.JPG')
            scores, boxes = im_detect(net, img_cv)

            height, width, c = img_cv.shape
            
            # bbs: '[x, y, w, h, num, leader]'
            bbs, bbs_other = utils.get_bbox(bbs_path, label, default_leader=leader)

            countData[0] += len(bbs) # 
            countData[1] += len(bbs_other)

            thresh = 0.8
            bbox = []
            for cls_ind, cls in enumerate(['face']):
                cls_ind += 1 # because we skipped background
                cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, 0.3)
                dets = dets[keep, :]
                inds = np.where(dets[:, -1] >= thresh)[0]
                for i in inds:
                    bbox.append(dets[i, :4])
            num += len(bbox)

            for i, d in enumerate(bbox):
                d = [int(round(d[0])), int(round(d[1])), int(round(d[2])), int(round(d[3]))]
                bound1 = (max(0, d[0]), max(0, d[1]),
                          min(d[2], width), min(d[3], height))

                img_temp = img_cv[max(0, bound1[1]):min(bound1[3], height),
                           max(0, bound1[0]):min(bound1[2], width)]
               
                f = model.get_feature_limited(img_temp)

                pad = 0
                if f is None:
                    # zoom in / out the rectangle if can't detect by MTCNN
                    # - : min(height, width) / 3
                    # + : max(height, width)
                    temp1 = min(bound1[3]-bound1[1], bound1[2]-bound1[0]) / 3
                    temp2 = max(bound1[3]-bound1[1], bound1[2]-bound1[0]) + 1

                    for j in range(-temp1, temp2):
                        # assure img_temp in img
                        if j == 0: continue
                        img_temp = img_cv[max(0, bound1[1] - j):min(bound1[3] + j, height),
                                   max(0, bound1[0] - j):min(bound1[2] + j, width)]
                        f = model.get_feature_limited(img_temp)
                        if f is not None:
                            pad = j
                            break
                bound1 = (max(0, bound1[0] - pad), max(0, bound1[1] - pad),
                          min(bound1[2]  + pad, width), min(bound1[3] + pad, height))
                
                cv2.rectangle(img_cv, (bound1[0], bound1[1]), (bound1[2], bound1[3]), (255, 0, 0))
                # cv2.imwrite(save_root+img_name+'.jpg', img_cv)
                # cv2.rectangle(img_cv, (bound1[0], bound1[1]), (bound1[2], bound1[3]), (255, 0, 0))
                # cv2.imshow('detect', img_cv)
                # cv2.waitKey(50)
                #
                if f is None:
                    # cv2.putText(img_cv, str(i)+'no detect', (bound1[0], bound1[1]), 0, 1, (255, 0, 0))
                    # cv2.imwrite(save_root+'no_detect/'+img_name+'.jpg', img_cv)
                    continue
                countDet += 1
                score = 999
                result = 'OTHERS'
                for feat_leader in feat_box:
                    for f2 in feat_box[feat_leader]:
                        dist = np.sum(np.square(f - f2))
                        
                        if dist < 0.9 and dist < score:
                            score = dist
                            result = feat_leader

                # flag = False 
                # if result != 'OTHERS':
                iou = 0
                positive = 'OTHERS'

                for bb in (bbs + bbs_other):
                    bound2 = (float(bb[0]), float(bb[1]),
                              float(bb[0]) + float(bb[2]), float(bb[1])+float(bb[3]))
                    iou = calculateIoU(bound1, bound2)
                    if iou >= 0.3:
                        positive = bb[5]
                        break
                if iou < 0.3:
                    print("the face no label")
                    # cv2.putText(img_cv, str(i) + 'no label', (bound1[0], bound1[1]), 0, 1, (255, 0, 0))
                    # cv2.imwrite(save_root +'no_label/' +img_name + '.jpg', img_cv)
                    no_label += 1
                    continue

                if result == positive: 
                    if result  == 'OTHERS':
                        right[1] += 1
                    else:
                        right[0] += 1
                elif positive != 'OTHERS': 
                    # cv2.putText(img_cv, str(i) + result  + ' ' + positive, (bound1[0], bound1[1]), 0, 0.5, (0, 255, 0))
                    # cv2.imwrite(save_root + 'falsePositive/' + img_name + '.jpg', img_cv)
                    FP += 1
                else: # result != 'OTHERS': 
                    FN += 1
                    # cv2.putText(img_cv, str(i) + result  + ' ' + positive, (bound1[0], bound1[1]), 0, 0.5, (0, 0, 255))
                    # cv2.imwrite(save_root + 'falseNegative/' + img_name + '.jpg', img_cv)
            # cv2.imwrite(save_root + img_name + '.jpg', img_cv)           

        output.write(' countData:{} leaders,{} others \n Sum:{} countDet:{} no_label:{}\n'.format( countData[0],countData[1], num, countDet, no_label )) 
        output.write('detection recall:{}\n'.format(num/float(countData[0]+countData[1])))
        precision  = float(right [0]+right [1])/(countDet - no_label)
        recall  = float(right [0])/countData[0]
        output.write('right:{},{} falsePositive:{} falseNegative:{}\n'.format(right [0], 
            right [1], FP , FN ))
        output.write('match precision :{}\n'.format(precision ))
        output.write('leader match recall:{}\n'.format(recall ))






