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
import face_align
import tensorflow as tf
from nets import nets_factory

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
    save_root = './false/result3'
    label_path_root = './result2'
    bbs_path_root = './txts'
    img_path_root = './raw'
    def BaseFeat(self, leader):
        feature = list()
        bases = glob.glob('raw2/' + leader + '-base/' +leader+'_*.jpg')
        # bases = glob.glob('raw2/' + leader + '-base/*.jpg')

        for n, path in enumerate(bases):
            label = os.path.basename(path)[:-4] + ';0;'
            print(label)
            img = cv2.imread(path)
            bbs, bbs_other = utils.get_bbox( 'txts/' + leader + '/', label, default_leader=leader)
            bound1 = (float(bbs[0][0]), float(bbs[0][1]), float(bbs[0][2])+float(bbs[0][0]), float(bbs[0][3])+float(bbs[0][1]))
            preds = fa.get_landmarks(path , bound1)[-1]
            ## face embedding 512d
            f, nimg = model.get_feature_by_landmark(img, bound1, preds)
            cv2.imwrite('raw2/{}-base/{}.jpg'.format(leader, n), nimg)
            
	    feature.append(f)
        return feature


tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', './cls_checkpoint/model.ckpt-60582',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
# tf.app.flags.DEFINE_string(
#     'test_list', '', 'Test image list.')
# tf.app.flags.DEFINE_string(
#     'test_dir', '.', 'Test image directory.')
# tf.app.flags.DEFINE_integer(
#     'batch_size', 16, 'Batch size.')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size')
FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    with tf.Graph().as_default():
        # tf_global_step = slim.get_or_create_global_step()

        ##################################
        # Load landmark detect and align #
        ##################################
        fa = face_align.FaceAlignment(face_align.LandmarksType._2D, enable_cuda=True)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes),
            is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name

        test_image_size = FLAGS.test_image_size or network_fn.default_image_size

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tensor_input = tf.placeholder(tf.float32, [1, test_image_size, test_image_size, 3])
        logits, _ = network_fn(tensor_input)
        logits = tf.nn.top_k(logits, 1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        ####################
        # Load InsightFace #
        ####################
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

        #################################
        # Load face detector model RFCN #
        #################################

        prototxt = '/home/disk3/py/py-R-FCN/models/pascal_voc/ResNet-101/rfcn_end2end/test_agonistic_face.prototxt'     
        caffemodel = '/home/disk3/py/py-R-FCN/data/rfcn_models/resnet101_rfcn_ohem_iter_40000.caffemodel'
        cfg.TEST.HAS_RPN = True
        caffe.set_mode_gpu()
        caffe.set_device(0)
        cfg.GPU_ID = 0
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        print ('\n\nLoaded network {:s}'.format(caffemodel))


        # leader feature
        feat_box = dict()
        cfg = Config()
        num = 0
        countDet = 0
        for leader in cfg.leaders_sml[:-1]:
            save_root = '{}/{}/'.format(cfg.save_root, leader)
            f1 = cfg.BaseFeat(leader)
            feat_box[leader] = f1
            if os.path.exists(save_root) is False:
                os.mkdir(save_root)
        exit(0)

        images_path = glob.glob('false/3.1/*.jpg')
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            for path in images_path:
                print(path)
                img_name = os.path.basename(path)
                img_cv = cv2.imread(path)
                if img_cv is None:
                    # errorImg_txt.write( img_name + " None\n")
                    continue

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
                print(num)
                for i, bound1 in enumerate(bbox):
                    preds = fa.get_landmarks(path, bound1)[-1]
                    ## face embedding 512d
                    f, _ = model.get_feature_by_landmark(img_cv, bound1, preds)

                    # d = [int(round(d[0])), int(round(d[1])), int(round(d[2])), int(round(d[3]))]
                    bound1 = (max(0, int(bound1[0])), max(0, int(bound1[1])),
                              min(int(bound1[2]), width), min(int(bound1[3]), height))
                    img_temp = img_cv[bound1[1]:bound1[3], bound1[0]:bound1[2]]
                    img_temp_cls = cv2.cvtColor(img_temp, cv2.COLOR_RGB2BGR)
                    img_temp_cls = cv2.resize(img_temp_cls, (299, 299), cv2.INTER_LINEAR)
                    img_temp_cls = np.array(img_temp_cls, dtype = np.uint8)
                    img_temp_cls = img_temp_cls.astype('float32')
                    img_temp_cls = np.multiply(img_temp_cls, 1.0/255.0)
                    img_temp_cls = np.subtract(img_temp_cls, 0.5)
                    img_temp_cls = np.multiply(img_temp_cls, 2.0)
                    img_temp_cls = np.expand_dims(img_temp_cls, 0)
                    predictions = sess.run(logits, feed_dict = {tensor_input : img_temp_cls}).indices
                    pose = predictions[0]
                    if pose == 0:
                        threshold = 1.1
                    if pose == 1:
                        threshold = 0.85

                    score = 999
                    result = 'OTHERS'
                    for feat_leader in feat_box:
                        for f2 in feat_box[feat_leader]:
                            dist = np.sum(np.square(f - f2))
                            print(feat_leader, dist)
                            if dist < threshold and dist < score:
                                score = dist
                                result = feat_leader
                    print(result, score)
                    exit(0)
                    if result != "OTHERS":
                        cv2.imwrite(cfg.save_root + '/' + result+ '/' + img_name[:-4] + '_' + str(i) +'.jpg', img_temp)
                        countDet += 1
                        print(countDet)




