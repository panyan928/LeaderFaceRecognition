from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,cv2
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import glob

slim = tf.contrib.slim



def get_bb(pic_bb_txt_lines,no):
    bbs = []
    for line in pic_bb_txt_lines:
        f = line.split(' ')
        if no == f[0]:
            bbs.append([f[1],f[2],f[3],f[4]])
    return bbs

def crop_pic(img_path,bb):
    print(img_path)
    if not os.path.isfile(img_path):
        img_path = img_path.split('.')[0] + '.JPG'

    image =cv2.imread(img_path)
    m,n,z = image.shape
    print(bb)
    #bb = bb[0]
    bbs = []
    l =len(bb)
    if l >0 and len(bb[0])>=4:
        for i in range(4):
            print(bb[0][i])
            m =float(bb[0][i])
            #print(m)s
            n = int(m)
            #print(n)
            bbs.append(n)
            #bbs.append(int(bb[0][i]))
        print(bbs)
        #image = image[int(bbs[1]):int(bbs[1])+int(bbs[3]),int(bbs[0]):int(bbs[0])+int(bbs[2])]
        x1 = bbs[0]
        y1 = bbs[1]
        x2 = bbs[0]+bbs[2]
        y2 = bbs[1]+bbs[3]
        if x1 <0:
            x1 = 1
        if y1 <0:
            y1 =1
        if x2 >=m:
            x2 =m-1
        if y2 >=n:
            y2 =n-1
        print(x1)
        print(x2)
        if y2 - y1 >2 and x2 - x1 >2:
            image = image[bbs[1]+1:bbs[1]+bbs[3]-1,bbs[0]+1:bbs[0]+bbs[2]-1]
        elif y2 - y1 >2 and x2 - x1 <=2:
            image = image[bbs[1]+1:bbs[1]+bbs[3]-1,bbs[0]:bbs[0]+bbs[2]]
        elif y2 - y1 <=2 and x2 - x1 >2:
            image = image[bbs[1]:bbs[1]+bbs[3],bbs[0]+1:bbs[0]+bbs[2]-1]
        else:
            image = image[bbs[1]:bbs[1]+bbs[3],bbs[0]:bbs[0]+bbs[2]]

        #image = image[int(y1):int(y2),int(x1):int(x2)]
        cv2.imwrite('pic_store/exchange/1.jpg',image)
        return image
    else:
        print('bb has no elemnet!')




txt_path = 'result2/'

bbs_path = 'txts/'

pic_path = 'raw/'

exchange_path = 'pic_store/exchange/'


pic_save_path = 'pic_store/pre/'

dirs = os.listdir(txt_path)

line_num = 0

learder_name_list = ['dxp','jzm','mzd','ply','xjp']

dict_learder_num = {'dxp':0,'jzm':0,'mzd':0,'ply':0,'xjp':0}



#for subdir in ['jzm']:
for subdir in learder_name_list:


    #print(subdir)

    subdir_U = subdir.upper()
    txt_path_f = txt_path + subdir_U + '/' +subdir_U + '.txt'

    pic_path_f = pic_path + subdir_U + '-craw/'

    print(txt_path_f)

    txt_pre = open(txt_path_f , 'r')

    learder_name = subdir.lower()

    print('learder_name : %s'%learder_name)

    for line in txt_pre.readlines():
        print(line)

        line=line.strip('\n')

        label_split = line.split(';') 

        pic_name = label_split[0]

        #len_txt = len(txt_pre)

        pic_bb_txt_path = bbs_path + subdir.upper() + '/' + pic_name + '_new' + '.txt'

        if os.path.isfile(pic_bb_txt_path):

            pic_bb_txt = open(pic_bb_txt_path,'r')

            pic_bb_txt_lines = pic_bb_txt.readlines()

            pic_path_full = pic_path_f + pic_name + '.jpg'



        l = len(label_split)
        #print(len(label_split))

        #if l ==2:
        #   print(l)

        #if l>1:
        #    pic_bb_txt_path = bbs_path + subdir.upper() + '/' + pic_name + '_new' + '.txt'

        if l == 1:
            print('no learder')
        #no subdir leader but has other leader
        elif l >=3 and len(label_split[1].split(':'))>1:
            print('no learder')

            for j in range(1,l-1):

                other = label_split[j].split(':')

                if len(other) == 1:
                    print('learder %s no %s'%(learder_name,other))
                    dict_learder_num[learder_name] +=1
                    bb = get_bb(pic_bb_txt_lines,other)

                    image=crop_pic(pic_path_full,bb)

                    print(other)


                    save_p = pic_save_path + learder_name + '/' + pic_name + '_' + str(other[0]) + '.jpg' 

                    cv2.imwrite(save_p,image)

                else:

                    other_no = label_split[j].split(':')[0]
                    other_name = label_split[j].split(':')[1]
                    if other_name not in learder_name_list:
                        print('other leader')
                        bb = get_bb(pic_bb_txt_lines,other_no)


                        image=crop_pic(pic_path_full,bb)

                        save_p = pic_save_path + 'other' + '/' + pic_name + '_' + other_no + '.jpg'

                        cv2.imwrite(save_p,image)

                    else:
                        dict_learder_num[other_name] +=1
                        print('learder %s no %s'%(other_name,other_no))
                        bb = get_bb(pic_bb_txt_lines,other_no)

                        image=crop_pic(pic_path_full,bb)

                        save_p = pic_save_path + other_name + '/' + pic_name + '_' + other_no + '.jpg'


                        cv2.imwrite(save_p,image)


        elif l ==3 and int(label_split[1]) == -1:
            print('no learder')
        elif l ==3 and int(label_split[1]) > -1:
            print('learder %s no %s'%(learder_name,label_split[1]))
            dict_learder_num[learder_name] +=1
            bb = get_bb(pic_bb_txt_lines,label_split[1])

            image=crop_pic(pic_path_full,bb)


            save_p = pic_save_path + learder_name + '/' + pic_name + '_' + label_split[1] + '.jpg' 

            cv2.imwrite(save_p,image)
        elif l >3 and len(label_split[1].split(':'))==1:

            if int(label_split[1]) == -1:
                print('no leader')
            elif int(label_split[1]) > -1:
                print('learder %s no %s'%(learder_name,label_split[1]))
                dict_learder_num[learder_name] +=1
                bb = get_bb(pic_bb_txt_lines,label_split[1])


                image=crop_pic(pic_path_full,bb)


                save_p = pic_save_path + learder_name + '/' + pic_name + '_' + label_split[1] + '.jpg' 

                cv2.imwrite(save_p,image)
            else:
                print('error')

            #if(label_split[2])


            for j in range(2,l-1):

                other = label_split[j].split(':')

                if len(other) == 1:
                    print('learder %s no %s'%(learder_name,other))
                    dict_learder_num[learder_name] +=1
                    bb = get_bb(pic_bb_txt_lines,other)

                    image=crop_pic(pic_path_full,bb)


                    save_p = pic_save_path + learder_name + '/' + pic_name + '_' + str(other[0]) + '.jpg' 

                    cv2.imwrite(save_p,image)
                else:

                    other_no = label_split[j].split(':')[0]
                    other_name = label_split[j].split(':')[1]
                    if other_name not in learder_name_list:
                        print('other leader')
                        bb = get_bb(pic_bb_txt_lines,other_no)

                        image=crop_pic(pic_path_full,bb)


                        save_p = pic_save_path + 'other' + '/' + pic_name + '_' + other_no + '.jpg' 

                        cv2.imwrite(save_p,image)
                    else:
                        dict_learder_num[other_name] +=1
                        print('learder %s no %s'%(other_name,other_no))
                        bb = get_bb(pic_bb_txt_lines,other_no)

                        image=crop_pic(pic_path_full,bb)


                        save_p = pic_save_path + other_name + '/' + pic_name + '_' + other_no + '.jpg' 

                        cv2.imwrite(save_p,image)


print(dict_learder_num.items())











