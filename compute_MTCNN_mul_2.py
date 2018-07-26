#!/usr/bin/env python

# -*- coding:utf-8 -*-
from skimage import io
import utils2 as utils
import glob, os
import numpy as np
import cv2
import face_embedding, argparse, shutil

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
            print(path)
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
    # leader feature
    feat_box = dict()
    cfg = Config()
    # boundingboxs numer of all leaders

    output = open('result_txt/result_MTCNN_mul_2.txt', 'w')
    output.write('total leaders: {}'.format(len(cfg.leaders_sml[:-1])))
    countDet = 0
    countData = [0, 0]
    no_label = 0
    right = dict()
    FN = dict()
    FP = dict() 
    precision = dict()
    recall = dict()
    score = dict()
    result = dict()
    num = 0
    true_profile = dict() #left face
    r = 0 #right face
    true_frontal = dict() #middle face
    false_profile = dict()
    false_frontal = dict()
    for leader in cfg.leaders_sml[:-1]:
    	save_root = '{}/{}/'.format(cfg.save_root, leader)
        f1 = cfg.BaseFeat(leader)
        feat_box[leader] = f1
    
	
    for threshold in range(60, 130, 5):
        threshold = threshold * 0.01
        right[threshold] = [0, 0]
        FN[threshold] = 0
        FP[threshold] = 0
        score[threshold] = 0
        precision[threshold] = 0
        recall[threshold] = 0
        true_frontal[threshold] = 0
        false_frontal[threshold] = 0
        true_profile[threshold] = 0
        false_profile[threshold] = 0
        result[threshold] = 'OTHERS'

    for leader in ['DXP']:

        output.write(leader)
        # result2/DXP/DXP.txt
        label_file_path = '{0}/{1}/{1}.txt'.format(cfg.label_path_root, leader)
        # txts/DXP/
        bbs_path = '{}/{}/'.format(cfg.bbs_path_root, leader)
        # raw/DXP-craw/
        img_path = '{}/{}-craw/'.format(cfg.img_path_root, leader)
        # detect_result/DXP/
        save_root = '{}/{}/'.format(cfg.save_root, leader)

        for label in open(label_file_path):
            if len(label) <= 11:
                continue
            print(label)
            # if int(label[6:9]) != 23: continue
            img_name = label.split(';')[0]
        
            img_cv = cv2.imread(img_path + img_name + '.jpg')
            if img_cv is None:
                img_cv = cv2.imread(img_path + img_name +'.JPG')
            boxes, points = model.detect(img_cv)
            if boxes is None:
                continue
            height, width, c = img_cv.shape
            
            # bbs: '[x, y, w, h, num, leader]'
            bbs, bbs_other = utils.get_bbox(bbs_path, label, default_leader=leader)

            countData[0] += len(bbs) # 
            countData[1] += len(bbs_other)
            num += len(boxes)

            for d, point in zip(boxes,points):
                f = model.get_feature(img_cv, d, point)
                bound1 = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
                countDet += 1
                for s in score:
                    score[s] = 999
                    result[s] = 'OTHERS'
                for feat_leader in feat_box:
                    for f2 in feat_box[feat_leader]:
                        dist = np.sum(np.square(f - f2))
                        for threshold in range(60, 130, 5):
                            threshold = threshold * 0.01
                            if dist < threshold and dist < score[threshold]:
                                score[threshold] = dist
                                result[threshold] = feat_leader

                # flag = False 
                # if result != 'OTHERS':
                iou = 0
                positive = 'OTHERS'

                for bb in (bbs + bbs_other):
                    bound2 = (float(bb[0]), float(bb[1]),
                              float(bb[0]) + float(bb[2]), float(bb[1])+float(bb[3]))
                    iou = calculateIoU(bound1, bound2)
                    if iou >= 0.5:
                        positive = bb[5]
                        break
                if iou < 0.5:
                    print("the face no label")
                    no_label += 1
                    continue
                for threshold in range(60, 130, 5):
                    threshold = threshold * 0.01
                    if result[threshold] == positive: 
                        if result[threshold] == 'OTHERS':
                            right[threshold][1] += 1
                        else:
                            right[threshold][0] += 1
                            if os.path.exists('/home/disk3/py/face_data/raw_split/DXP/'+img_name+'.jpg'):
                                true_frontal[threshold] +=1
                            elif os.path.exists('/home/disk3/py/face_data/raw_split/DXP/'+img_name+'.JPG'):
                                true_frontal[threshold] +=1
                            else:
                                true_profile[threshold] += 1
                    elif positive != 'OTHERS': 
                        # cv2.putText(img_cv, str(i) + result[threshold] + ' ' + positive, (bound1[0], bound1[1]), 0, 0.5, (0, 255, 0))
                        # cv2.imwrite(save_root + 'falsePositive/' + img_name + '.jpg', img_cv)
                        FP[threshold] += 1
                        if os.path.exists('/home/disk3/py/face_data/raw_split/DXP/'+img_name+'.jpg'):
                            false_frontal[threshold] +=1
                        elif os.path.exists('/home/disk3/py/face_data/raw_split/DXP/'+img_name+'.JPG'):
                            false_frontal[threshold] +=1
                        else:
                            false_profile[threshold] += 1
                    else: # result != 'OTHERS': 
                        FN[threshold]+= 1
                        # cv2.putText(img_cv, str(i) + result[threshold] + ' ' + positive, (bound1[0], bound1[1]), 0, 0.5, (0, 0, 255))
                        # cv2.imwrite(save_root + 'falseNegative/' + img_name + '.jpg', img_cv)
            # cv2.imwrite(save_root + img_name + '.jpg', img_cv)


        output.write(' countData:{} leaders,{} others \n Sum:{} countDet:{} no_label:{}\n'.format( countData[0],countData[1], num, countDet, no_label )) 
        output.write('detection recall:{}\n'.format(num/10088))
        for threshold in range(60, 130, 5):
            threshold = threshold * 0.01
            precision[threshold] = float(right[threshold][0]+right[threshold][1])/(countDet - no_label)
            recall[threshold] = float(right[threshold][0])/countData[0]
            output.write('threshold:{}\n'.format(threshold))
            output.write('right:{},{} falsePositive:{} falseNegative:{}\n'.format(right[threshold][0], 
                right[threshold][1], FP[threshold], FN[threshold]))
            output.write('match precision :{}\n'.format(precision[threshold]))
            output.write('leader match recall:{}\n'.format(recall[threshold]))
            output.write('true profile:{} false profile:{}\n'.format(true_profile[threshold], false_profile[threshold]))
            output.write('true frontal:{} false frontal:{}\n'.format(true_frontal[threshold], false_frontal[threshold]))
    # print(' {} labeled leaders, {} others '.format(countData[0],countData[1]))
    # print(' all test faces: {}, detected and got feature'.format(countDet))
    # print('precision :{}'.format(right/(countDet-no_label)))
    # print('recall:{}'.f`ormat(right[0]/countData[0]))
    # print('no_label:'+ str(no_label))
    # print('falsePositive:' + str(FP))
    # print('falseNegative:' + str(FN))






