#!/usr/bin/env python

# -*- coding:utf-8 -*-

import dlib
from skimage import io
import utils2 as utils
import glob, os
import numpy as np
import cv2
import face_embedding, argparse

def calculateIoU(candidateBound, gintTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = gintTruthBound[0]
    gy1 = gintTruthBound[1]
    gx2 = gintTruthBound[2]
    gy2 = gintTruthBound[3]

    carea = (cx2 - cx1) * (cy2 - cy1) #
    garea = (gx2 - gx1) * (gy2 - gy1) #

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
    leaders_sml = ['DXP','JZM', 'MZD', 'PLY', 'XJP','OTHERS']#,'JZM', 'MZD', 'PLY', 'XJP',
    save_root = './detect_compute'
    label_path_root = './result2'
    bbs_path_root = './txts'
    img_path_root = './raw'
    def BaseFeat(self, leader):
        feature = list()
        bases = glob.glob('raw/' + leader + '-base/*.jpg')
        for path in bases:
            # print(path)
            img = cv2.imread(path)
            label = os.path.basename(path)[:-4] + ';0;'
            if len(label)<6:
                continue
            bbs, bbs_other = utils.get_bbox('txts/' + leader + '/', label, default_leader=leader)
            for i in [0] + list(range(20, 102, 2)):
                i = i*0.1
                if i==0:
                    pad = [0,0]
                else:
                    pad = [float(bbs[0][2]) / i, float(bbs[0][3]) / i]
                bbs_temp = (max(0, int(float(bbs[0][0]) - pad[0])), max(0, int(float(bbs[0][1]) - pad[1])),
                            min(img.shape[1], int(float(bbs[0][0]) + float(bbs[0][2]) + pad[0])),
                            min(img.shape[0], int(float(bbs[0][1]) + float(bbs[0][3]) + pad[1])))
                img_temp = img[bbs_temp[1]:bbs_temp[3], bbs_temp[0]:bbs_temp[2]]
                # cv2.imshow("crop", img_temp)
                # cv2.waitKey(0)
                f = model.get_feature(img_temp)
                if f is not None:
                    break
            if f is None: print(path)
            print(path)
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
    
    feat_box = dict()
    # 
    detector = dlib.get_frontal_face_detector()
    cfg = Config()
    # boundingboxs numer of all leaders
    output = open('result-DXP-dlib.txt', 'w')
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
    print('total leaders: {}'.format(len(cfg.leaders_sml[:-1])))

    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for leader in cfg.leaders_sml[:-1]:
        f1 = cfg.BaseFeat(leader)
        feat_box[leader] = f1

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
            # label = 'DXP_00166;0;'
            if len(label) <= 10:
                continue
            print(label)
            # if int(label[6:9]) != 23: continue
            img_name = label.split(';')[0]

            try:
                img = io.imread(img_path + img_name + '.jpg')
            except IOError:
                img = io.imread(img_path + img_name + '.JPG')

            img_cv = cv2.imread(img_path + img_name + '.jpg')
            # #print(img.shape)
            # if len(img.shape) != 3 or img.shape[2] == 4:
            #     print('4 channel image or gray image')
            #     cv2.imwrite(img_path + img_name + '.jpg', img_cv)
            #     img = io.imread(img_path + img_name + '.jpg')
            if img_cv is None:
                img_cv = cv2.imread(img_path + img_name +'.JPG')
            height, width, c = img_cv.shape
            # all faces in image
            dets = detector(img, 2)
            # bbs: '[x, y, w, h, num, leader]'
            bbs, bbs_other = utils.get_bbox(bbs_path, label, default_leader=leader)
            countData[0] += len(bbs) # 
            countData[1] += len(bbs_other)
            # if len(dets) == 0:
                # cv2.imwrite(save_root + img_name + '.jpg', img_cv)
                # print("can't detct face")
            num += len(dets)
            for i, d in enumerate(dets):
                bound1 = (max(0, d.left()), max(0, d.top()),
                          min(d.right(), width), min(d.bottom(), height))

                img_temp = img_cv[max(0, bound1[1]):min(bound1[3], height),
                           max(0, bound1[0]):min(bound1[2], width)]
                # 
                f = model.get_feature(img_temp)
                # 
                pad = [0, 0]
                if f is None:
                    for j in [-8, -6, -4, 4, 6, 8]:
                        pad = [int((d.right()-d.left()) / j), int((d.bottom()-d.top()) / j)]
                        # assure 
                        img_temp = img_cv[max(0, bound1[1]+pad[1]):min(bound1[3]-pad[1], height),
                                   max(0, bound1[0]+pad[0]):min(bound1[2]-pad[0], width)]
                        f = model.get_feature(img_temp)
                        if f is not None:
                            break
                bound1 = (max(0, bound1[0] + pad[0]), max(0, bound1[1] + pad[1]),
                          min(bound1[2] - pad[0], width), min(bound1[3] - pad[1], height))
                cv2.rectangle(img_cv, (bound1[0], bound1[1]), (bound1[2], bound1[3]), (255, 0, 0))
                # cv2.imshow('result', img_cv)
                # cv2.waitKey(50)
                # 
                if f is None:
                    #cv2.putText(img_cv, str(i)+'no detect', (bound1[0], bound1[1]), 0, 1, (255, 0, 0))
                    #cv2.imwrite(save_root + img_name +'.jpg', img_cv)
                    continue
                countDet += 1# 
                # 
                score = 999
                result = 'OTHERS'
                for feat_leader in feat_box:
                    
                    flag = False #
                    for f2 in feat_box[feat_leader]:
                        dist = np.sum(np.square(f - f2))
                        # print(dist)
                        # dist = np.sum(np.square(f - f2))
                     

                        if dist < 1.2 and dist < score:
                            score = dist
                            result = feat_leader

                # 
                # if result != 'OTHERS':
                iou = 0
                positive = 'OTHERS'

                # cv2.putText(img_cv,str(score),(bound1[0], bound1[1]), 0, 0.5, (255, 0, 0))
                # cv2.imwrite(save_root + img_name + '.jpg', img_cv)
                # cv2.imshow('result', img_cv)
                # cv2.waitKey(0)
                # find a ginttruth boundingbox and positive leader
                for bb in (bbs + bbs_other):
                    bound2 = (round(float(bb[0])), round(float(bb[1])),
                              round(float(bb[0]) + float(bb[2])), round(float(bb[1])+float(bb[3])))
                    iou = calculateIoU(bound1, bound2)
                    if iou > 0.5:
                        positive = bb[5]
                        break

                if iou < 0.5:
                    print("the face no label")
                    # cv2.putText(img_cv, str(i) + result + 'no label', (bound1[0], bound1[1]), 0, 0.5, (255, 0, 0))
                    # cv2.imwrite(save_root +'no_label/' +img_name + '.jpg', img_cv)
                    no_label += 1
                    break

                if result == positive:# 
                    if result == 'OTHERS':
                        right[1] += 1
                    else:
                        right[0] += 1

                elif positive != 'OTHERS':
                    # 
                    # cv2.putText(img_cv, str(i) + result + ' ' + positive, (bound1[0], bound1[1]), 0, 0.5, (0, 255, 0))
                    # cv2.imwrite(save_root + 'falsePositive/' + img_name + '.jpg', img_cv)
                    FP += 1
                else:
                    # result[threshold] != 'OTHERS':
                    # #
                    FN += 1
                    # cv2.putText(img_cv, str(i) + result + ' ' + positive, (bound1[0], bound1[1]), 0, 0.5, (0, 0, 255))
                    # cv2.imwrite(save_root + 'falseNegative/' + img_name + '.jpg', img_cv)

        print(' {} labeled leaders, {} others\n'.format(countData[0], countData[1]))
        print(' {} detected faces, {} tested faces(detected and got feature)\n'.format(num, countDet))
        print(' right:{},{} no_label:{} falsePositive:{} falseNegative:{}\n'.format(
            right[0], right[1], no_label, FP, FN))
        precision  = float(right [0]+right [1])/(countDet - no_label)
        recall  = float(right [0])/countData[0]
        print(' match precision :{}\n'.format(precision))
        print('leader match recall:{}\n'.format(recall ))
        print('detection recall:{}\n'.format(num/float(countData[0]+countData[1])))




