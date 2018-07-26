#!//usr/bin/env python

# -*- coding:utf-8 -*-

from skimage import io
import utils2 as utils
import glob, os
import numpy as np
import cv2
import face_embedding, argparse, shutil

class Config:
    leaders_sml = ['DXP','JZM', 'MZD', 'PLY', 'XJP', 'OTHERS']#,
    save_root = './txt2'
    label_path_root = './result2'
    bbs_path_root = './txts'
    img_path_root = './raw'

if __name__ == '__main__':
    cfg = Config()
    for leader in cfg.leaders_sml[:-1]:
        # result2/DXP/DXP.txt
        label_file_path = '{0}/{1}/{1}_new.txt'.format(cfg.label_path_root, leader)
        # txts/DXP/
        bbs_path = '{}/{}/'.format(cfg.bbs_path_root, leader)
        # raw/DXP-craw/
        img_path = '{}/{}-craw/'.format(cfg.img_path_root, leader)
        # detect_result/DXP/
        save_root = '{}/{}/'.format(cfg.save_root, leader)

