#!/usr/bin/env python

# -*- coding:utf-8 -*-
import cv2
from PIL import Image,ImageDraw
import os
def get_bbox(bbs_path, label, default_leader):
    bbs = list()
    bbs_other= list()
    # print(label)
    name = label.split(';')[0]
    txt_path = bbs_path + name + '.txt'
    if not os.path.exists(txt_path):
        return None
    # if os.path.exists(txt_path) is False:
    #     txt_path = bbs_path + label.split(';')[0] + '.txt'
    for line in open(txt_path):
        info = line.split(' ')
        if len(info) != 7:
            print('error line in {}: {}'.format(txt_path, info))
            continue
        flag = False
        temp = label.split(';')
        pose = -1
        for i in temp[1:]:
            # label type: '1'  '11'  '1:xjp' '-1'
            if len(i)<=2 and info[0] == i:
                bbs.append([info[1], info[2], info[3], info[4], info[0], default_leader, pose])
                flag = True
                break
            elif len(i)>2 and info[0]==i.split(':')[0]:
                other_leader = i.split(':')[1]
                if other_leader == 'xjp' or other_leader=='dxp' or other_leader=='mzd' or other_leader=='ply' or other_leader == 'jzm':
                    flag = True
                    bbs.append([info[1], info[2], info[3], info[4], info[0], other_leader.upper(), pose])
        if not flag:
            bbs_other.append([info[1], info[2], info[3], info[4], info[0], 'OTHERS', pose])
    return bbs, bbs_other

def get_img_path(img_path, img_name):
    return img_path+img_name+'.jpg'

#Image type
def write_bbs_Image(img, bbs, save_root, img_name, flag):
    if flag == 1:
        color = '#FF0000'
    else:
        color = None
    for bb in bbs:
        draw = ImageDraw.Draw(img)
        # print(type(bb[1]))
        draw.rectangle([(round(float(bb[0])), round(float(bb[1]))),
                        round(float(bb[0])+ float(bb[2])), round(float(bb[1])+float(bb[3]))],
                       outline=color)
        draw.text((round(float(bb[0])), round(float(bb[1]))), bb[4])
    # img.show()
    print(save_root + img_name+'.jpg')
    # img.save(save_root + img_name+'.jpg', 'JPEG')
#opencv
def write_bbs(img, bbs, save_root, img_name, flag):
    # if flag == 1:
    #     color = '#FF0000'
    # else:
    #     color = None
    for bb in bbs:
        cv2.rectangle(img, (round(float(bb[0])),round(float(bb[1]))),
                      (round(float(bb[0]) + float(bb[2])), round(float(bb[1]) + float(bb[3]))),
                      (255,255,255), 1)
        cv2.putText(img, bb[5], (round(float(bb[0])),round(float(bb[1]))), fontScale=1, fontFace= 1, color=(255,255,255))
        # draw = ImageDraw.Draw(img)
    cv2.imshow("image",img)
    # print(save_root + img_name+'.jpg')
    # img.write(save_root + img_name+'.jpg', 'JPEG')

def cropped_img(img, bbs):
    cropped = list()
    for bb in bbs:
        # print(bb)
        img_crop = img[max(0,round(float(bb[1]))):round(float(bb[1])+float(bb[3])),
                   max(0, round(float(bb[0]))):round(float(bb[0])+float(bb[2]))]
        cropped.append(img_crop)
    return cropped