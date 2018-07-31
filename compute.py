#!/usr/bin/env python

import _init_paths

import os, glob, cv2, sys
import numpy as np
import tensorflow as tf
from nets import nets_factory
import utils2 as utils
import face_embedding, argparse
import _init_paths
from fast_rcnn.config import cfg as detect_cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import caffe
import face_align
import mxnet as mx

slim = tf.contrib.slim

def parse_args():
    parser = argparse.ArgumentParser(description='Compute Recognition Result')
    parser.add_argument('--result', default = 'compute', help = '')
    parser.add_argument('--output-txt', default='result_txt/result.txt', help = 'file to save computation result')
    parser.add_argument('--detection-model', default='RFCN', help = 'detection model, RFCN, MTCNN or DLIB')
    parser.add_argument('--landmarks-model', default= 'FaceAlign', help = 'get landmarks model, FaceAlign or MTCNN')
    parser.add_argument('--mul-thres', type = bool, default = True, help = 'whether use multi thresholds')
    parser.add_argument('--thresholds', type = str, default = '0.8, 1.6')
    parser.add_argument('--save-root', default = './detect_compute')
    parser.add_argument('--gpu', type = int, default = 1)
    args = parser.parse_args()
    return args

def init_dict(min_threshold, max_threshold, list=False, str = False):
    temp = dict()
    if list:
        for threshold in range(int(min_threshold * 100), int(max_threshold * 100), 5):
            temp[threshold] = [0, 0]
        return temp
    if str:
        for threshold in range(int(min_threshold * 100), int(max_threshold * 100), 5):
            temp[threshold] = 'OTHERS'
        return temp
    for threshold in range(int(min_threshold * 100), int(max_threshold * 100), 5):
        temp[threshold] = 0
    return temp


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
    def __init__(self):
        self.leaders_sml = ['DXP', 'JZM', 'MZD', 'PLY', 'XJP', 'OTHERS']
        self.label_path_root = './result2'
        self.bbs_path_root = './txts_new'
        self.img_path_root = './raw'
        self.feat_box_fro = dict()
        self.feat_box_pro = dict()
        self.img_box_fro = dict()
        self.img_box_pro = dict()

    def BaseFeat(self, leader):
        feature_fro = list()
        feature_pro = list()
        img_fro = list()
        img_pro = list()
        bases = glob.glob('raw3/' + leader + '-base/' +leader+'_*.jpg')
        # bases = glob.glob('raw2/' + leader + '-base/*.jpg')
        for path in bases:
            print(path)
            img = cv2.imread(path)
            bbs, bbs_other = utils.get_bbox_new( self.bbs_path_root +'/'+ leader + '/', os.path.basename(path)[:-4])
            bound1 = (float(bbs[0][0]), float(bbs[0][1]), float(bbs[0][2])+float(bbs[0][0]), float(bbs[0][3])+float(bbs[0][1]))
            preds = fa.get_landmarks(path, bound1)[-1]
            ## face embedding 512d
            f, nimg = model.get_feature_by_landmark(img, bound1, preds)
            if bbs[0][6] == '1':
                feature_pro.append(f)
                img_pro.append(nimg)
            elif bbs[0][6] == '0' or bbs[0][6] == '2':
                feature_fro.append(f)
                img_fro.append(nimg)
            else:
                print('no profile / frontal file\n')
        return feature_fro, img_fro, feature_pro, img_pro
    def set(self):
        for leader in self.leaders_sml[:-1]:
            self.feat_box_fro[leader], self.img_box_fro[leader], self.feat_box_pro[leader], \
            self.img_box_pro = self.BaseFeat(leader)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)

    args = parse_args()

    #################################
    # Load face detector model RFCN #
    #################################
    if args.detection_model == 'RFCN':
        prototxt = './py-R-FCN/model/test_agonistic_face.prototxt'
        caffemodel = './py-R-FCN/model/resnet101_rfcn_ohem_iter_40000.caffemodel'
        detect_cfg.TEST.HAS_RPN = True
        caffe.set_mode_gpu()
        caffe.set_device(0)
        detect_cfg.GPU_ID = 0
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        print ('\n\nLoaded network {:s}'.format(caffemodel))
    ##################################
    # Load landmark detect and align #
    ##################################
    if args.landmarks_model == 'FaceAlign':
        fa = face_align.FaceAlignment(face_align.LandmarksType._2D, enable_cuda=True)

    ####################
    # Load InsightFace #
    ####################
    parser_i = argparse.ArgumentParser(description='face model test')
    parser_i.add_argument('--image-size', default='112,112', help='')
    parser_i.add_argument('--model', default='./model-r50-am-lfw/model,0', help='path to load model.')
    parser_i.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser_i.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
    parser_i.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    args_i = parser_i.parse_args()
    model = face_embedding.FaceModel(args_i)

    ######################
    # Load Vec2Vec Model #
    ######################

    sym, arg_params, aux_params = mx.model.load_checkpoint('model-vec2vec/model-leader-xjp', 48)
    mod_vec2vec = mx.mod.Module(sym, context=mx.gpu(0), label_names=None)
    mod_vec2vec.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 112))], label_shapes=mod_vec2vec._label_shapes)
    mod_vec2vec.set_params(arg_params, aux_params, allow_missing=True)

    with tf.Graph().as_default():
        # tf_global_step = slim.get_or_create_global_step()
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

        ###############################
        # Deploy file path and params #
        ###############################
        cfg = Config()
        cfg.set()
        output = open(args.output_txt, 'w')
        output.write('total leaders: {}'.format(len(cfg.leaders_sml[:-1])))
        countDet = 0
        countData = [0, 0]
        no_label = 0

        if args.mul_thres:
            assert ',' in args.thresholds
            min_thre = float(args.thresholds.split(',')[0])
            max_thre = float(args.thresholds.split(',')[1])
            right_fro = init_dict(min_thre, max_thre, list=True)
            score = init_dict(min_thre, max_thre)
            result = init_dict(min_thre, max_thre, str = True)
            num = 0
            fp_fro = init_dict(min_thre, max_thre)
            fn_fro = init_dict(min_thre, max_thre)
        else:
            threshold = float(args.thresholds)
            right_fro = [0, 0]
            num = 0
            fp_fro = 0
            fn_fro = 0
        right_pro = [0, 0]
        fn_pro = 0
        fp_pro = 0

        for leader in cfg.leaders_sml[:-1]:
            save_root = '{}/{}/'.format(args.save_root, leader)
            if not os.path.exists(save_root):
                os.mkdir(save_root)
            if not os.path.exists(save_root + 'right0/'):
                os.mkdir(save_root + 'right0/')
                os.mkdir(save_root + 'right1/')
                os.mkdir(save_root + 'falseNegative')
                os.mkdir(save_root + 'falsePositive')
                os.mkdir(save_root + 'no_label')

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, cls_checkpoint_path)

            ###############################
            # Detect image Loop by leader #
            ###############################
            for leader in ['XJP']:
                output.write(leader)
                # result2/DXP/DXP.txt
                label_file_path = '{0}/{1}/{1}.txt'.format(cfg.label_path_root, leader)
                # txts_new/DXP/
                bbs_path = '{}/{}/'.format(cfg.bbs_path_root, leader)
                # raw/DXP-craw/
                img_path = '{}/{}-craw/'.format(cfg.img_path_root, leader)
                # detect_result/DXP/
                save_root = '{}/{}/'.format(args.save_root, leader)
                img_names = os.listdir(img_path)
                img_names.sort()
                for img_name in img_names:
                    print(img_name)
                    # load image
                    img_cv = cv2.imread(img_path + img_name)
                    # img_tf = open(img_path + img_name, 'rb').read()
                    # img_tf = tf.image.decode_jpeg(img_tf, channels=3)

                    # Get groundtruth of one image 
                    # bbs: '[x, y, w, h, num, leader]'
                    gt = utils.get_bbox_new(bbs_path, img_name)
                    if gt is None:
                        raise RuntimeError
                    bbs = gt[0]
                    bbs_other = gt[1]
                    countData[0] += len(bbs) # 
                    countData[1] += len(bbs_other)
                    # new_bbs_txt = open(bbs_path + img_name[:-4] + '_new.txt', 'w')

                    ###############################
                    # Detect all faces in a image #
                    ###############################
                    scores, boxes = im_detect(net, img_cv)
                    height, width, c = img_cv.shape
                    thresh = 0.9
                    bbox = []
                    for cls_ind, cls in enumerate(['face']):
                        cls_ind += 1 # because we skipped background
                        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                        cls_scores = scores[:, cls_ind]
                        dets = np.hstack((cls_boxes,
                                          cls_scores[:, np.newaxis])).astype(np.float32)
                        keep = nms(dets, 0.5)
                        dets = dets[keep, :]
                        inds = np.where(dets[:, -1] >= thresh)[0]
                        for i in inds:
                            bbox.append(dets[i, :4])
                    num += len(bbox)

                    for d in bbox:
                        # first find groundtruth, 
                        iou = 0
                        positive = 'OTHERS'
                        pose = -1
                        for bb in (bbs + bbs_other):
                            bound2 = (float(bb[0]), float(bb[1]),
                                      float(bb[0]) + float(bb[2]), float(bb[1])+float(bb[3]))
                            iou = calculateIoU(d, bound2)
                            if iou >= 0.5:
                                positive = bb[5]
                                pose = bb[6]
                                break
                        # no groundtruth , continue to next face
                        if iou < 0.5:
                            print("the face no label")
                            if not args.mul_thres:
                                cv2.putText(img_cv, str(i) + 'no label', (int(d[0]), int(d[1])), 0, 1, (255, 0, 0))
                                cv2.imwrite(save_root +'no_label/' + img_name, img_cv)
                            no_label += 1
                            continue

                        countDet += 1
                        # d = [int(round(d[0])), int(round(d[1])), int(round(d[2])), int(round(d[3]))]
                        bound1 = (max(0, int(d[0])), max(0, int(d[1])),
                                  min(int(d[2]), width), min(int(d[3]), height))

                        ### Add a step
                        ### classify the face profile or frontal
                        ### groundtruth no pose information, classify by netwrok
                        if pose == -1:
                            img_temp_cls = img_cv[bound1[1]:bound1[3], bound1[0]:bound1[2]]
                            img_temp_cls = cv2.cvtColor(img_temp_cls, cv2.COLOR_RGB2BGR)
                            img_temp_cls = cv2.resize(img_temp_cls, (299, 299), cv2.INTER_LINEAR)
                            img_temp_cls = np.array(img_temp_cls, dtype = np.uint8)
                            img_temp_cls = img_temp_cls.astype('float32')
                            img_temp_cls = np.multiply(img_temp_cls, 1.0/255.0)
                            img_temp_cls = np.subtract(img_temp_cls, 0.5)
                            img_temp_cls = np.multiply(img_temp_cls, 2.0)
                            img_temp_cls = np.expand_dims(img_temp_cls, 0)
                            predictions = sess.run(logits, feed_dict = {tensor_input : img_temp_cls}).indices
                            pose = predictions[0]

                        ## face landmark detection
                        preds = fa.get_landmarks(img_path + img_name, d)[-1]
                        ## face embedding 512d
                        f, nimg = model.get_feature_by_landmark(img_cv, d, preds)

                        if pose == 1: ## use vec2vec model when assured profiles
                            max_prob = 0
                            result_v2v = 'OTHERS'
                            for leader in cfg.leaders_sml[:-1]:
                                for f_img in cfg.img_box_fro[leader]:
                                    img_contate = np.concatenate(nimg, f_img, axis = 0)
                                    img_contate = img_contate.transpose((2,0,1))
                                    img_contate = np.expand_dims(img_contate, 0)
                                    test = mx.nd.array(img_contate)
                                    test = mx.io.DataBatch(data=(test,))
                                    mod_vec2vec.forward(test)
                                    prob = mod_vec2vec.get_outputs()[0].asnumpy()
                                    prob = np.squeeze(prob)
                                    a = np.argsort(prob)[::-1]
                                    ### vec2vec model predict possibility/probability of same person
                                    if a[0] == 0 and max_prob < prob[0]:
                                        result_v2v = leader
                                        max_prob = prob[0]
                            ## compute right or wrong
                            if result_v2v == positive:
                                if positive != 'OTHERS':
                                    right_pro[0] += 1
                                    cv2.imwrite('{}right0/{}_p_{}.jpg'.format(save_root, img_name[:-4], i), nimg)
                                else:
                                    right_pro[1] += 1
                                    cv2.imwrite('{}right1/{}_p_{}.jpg'.format(save_root, img_name[:-4], i), nimg)
                            elif positive != 'OTHERS':
                                fn_pro += 1
                            else:
                                fp_pro += 1

                        else:
                            feat_box = cfg.feat_box_fro
                            if args.mul_thres:
                                for s in score:
                                    score[s] = 999
                                    result[s] = 'OTHERS'
                                for feat_leader in feat_box:
                                    for f2 in feat_box[feat_leader]:
                                        dist = np.sum(np.square(f - f2))
                                        for threshold in score:
                                            if dist < threshold and dist < score[threshold]:
                                                score[threshold] = dist
                                                result[threshold] = feat_leader
                        
                                for threshold in result:
                                    if result[threshold] == positive:
                                        if result[threshold] == 'OTHERS':
                                            right_fro[threshold][1] += 1
                                        else:
                                            right_fro[threshold][0] += 1
                                    elif positive != 'OTHERS':
                                        fn_fro[threshold] += 1
                                    else: # result != 'OTHERS':
                                        fp_fro[threshold] += 1

                                # img_temp = img_cv[bound1[1]:bound1[3], bound1[0]:bound1[2]]
                                # print(bound1)
                                # run classification
                    output.write('profile:\n')
                    output.write('right:{},{} fp:{}, fn:{}\n'.format(right_pro[0], right_pro[1],
                                                                     fp_pro, fn_pro))
                    for threshold in right_fro:
                        output.write('threshold:{}\n'.format(threshold))
                        output.write('frontal:\n')
                        output.write(
                            'right:{},{} fp:{}, fn:{}\n'.format(right_fro[threshold][0], right_fro[threshold][1],
                                                                fp_fro[threshold], fn_fro[threshold]))

                ## countData: groundtruth sum
                ## sum : all detection result  == countDet + no_label

                output.write('countData:{} leaders,{} others \n Sum:{} countDet:{} no_label:{}\n'.format( countData[0],countData[1], num, countDet, no_label ))
                output.write('detection recall:{}\n'.format(countDet/float(countData[0]+countData[1])))
                output.write('profile:\n')
                output.write('right:{},{} fp:{}, fn:{}\n'.format(right_pro[0], right_pro[1],
                                                                 fp_pro, fn_pro))

                for threshold in score:
                    if countDet != 0:
                        precision = float(right_fro[threshold][0]+right_fro[threshold][1] + right_pro[0] +right_pro[1]) / (countDet)
                    else:
                        precision = 0
                    right = [right_fro[threshold][0]+ right_pro[0], right_fro[threshold][1]+right_pro[1]]
                    falsepositive = fp_fro[threshold]+fp_pro
                    falsenegative = fn_fro[threshold]+fn_pro
                    precision = float(right)/(right+ falsepositive)
                    recall = float(right) / (right + falsenegative)

                    output.write('threshold:{}\n'.format(threshold))
                    output.write('right:{},{} falsePositive:{} falseNegative:{}\n'.format(right[0],
                        right[1], falsepositive, falsenegative))
                    output.write('precision :{}\n'.format(precision))
                    output.write('recall:{}\n'.format(recall))
                    output.write('frontal:\n')
                    output.write('right:{},{} fp:{}, fn:{}\n'.format(right_fro[threshold][0],right_fro[threshold][1], fp_fro[threshold], fn_fro[threshold]))

