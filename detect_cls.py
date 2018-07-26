from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,json,math,time,glob,cv2
import numpy as np
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
from skimage import io
import utils2 as utils
import face_embedding, argparse, shutil
import _init_paths
from fast_rcnn.config import cfg as detect_cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import caffe
import matplotlib.pyplot as plt
import gc

slim = tf.contrib.slim
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
    save_root = './detect_cls'
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



tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', './cls_checkpoint',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
# tf.app.flags.DEFINE_string(
#     'test_list', '', 'Test image list.')
# tf.app.flags.DEFINE_string(
#     'test_dir', '.', 'Test image directory.')
# tf.app.flags.DEFINE_integer(
#     'batch_size', 16, 'Batch size.')
tf.app.flags.DEFINE_integer(
    'num_classes', 3, 'Number of classes.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size')
FLAGS = tf.app.flags.FLAGS

def main(_):
    # tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # tf_global_step = slim.get_or_create_global_step()

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
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
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
        # test_ids = [line.strip() for line in open(FLAGS.test_list)]
        # tot = len(test_ids)
        # results = list()

        ###############################
        # Deploy file path and params #
        ###############################
        cfg = Config()
        feat_box = dict()
        countDet = 0
        countData = [0, 0]
        no_label = 0
        right = [0, 0]
        FN = 0
        FP = 0
        precision = 0
        recall = 0
        num = 0

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            ###############################
            # Detect image Loop by leader #
            ###############################
            for leader in cfg.leaders_sml[:-1]:
            	# leader = 'DXP'
                # output.write(leader)
                # result2/DXP/DXP.txt
                label_file_path = '{0}/{1}/{1}.txt'.format(cfg.label_path_root, leader)
                # txts/DXP/
                bbs_path = '{}/{}/'.format(cfg.bbs_path_root, leader)
                # raw/DXP-craw/
                img_path = '{}/{}-craw/'.format(cfg.img_path_root, leader)
                # detect_result/DXP/
                save_root = '{}/{}/'.format(cfg.save_root, leader)

                right = 0
                wrong = 0
                for label in open(label_file_path):
                    if len(label) <= 10:
                        continue
                    # if int(label[5:9]) > 1583 and int(label[5:9]) <= 1760: continue
                    img_name = label.split(';')[0]+'.jpg'
                    # print('raw2/' + leader + '-base/' + img_name)
                    # if os.path.exists('raw2/' + leader + '-base/' + img_name) is False: continue
                    if os.path.exists(bbs_path + img_name[:-4] + '_new.txt'): continue
                    print(img_name)
                    # load image
                    img_cv = cv2.imread(img_path + img_name)
                    if img_cv is None:
                        img_name = img_name[:-4]+'.JPG'
                        img_cv = cv2.imread(img_path + img_name)
                    # img_tf = open(img_path + img_name, 'rb').read()
                    # img_tf = tf.image.decode_jpeg(img_tf, channels=3)
                    ###############################
                    # Detect all faces in a image #
                    ###############################
                    height, width, c = img_cv.shape

                    # print(num)
                    # Get groundtruth of one image 
                    # bbs: '[x, y, w, h, num, leader]'
                    bbs, bbs_other = utils.get_bbox(bbs_path, label, default_leader=leader)
                    countData[0] += len(bbs) # 
                    countData[1] += len(bbs_other)
                    new_bbs_txt = open(bbs_path + img_name[:-4] + '_new.txt', 'w')

                    for bb in (bbs + bbs_other):

                        d = (int(float(bb[0])), int(float(bb[1])),
                                      int(float(bb[0]) + float(bb[2])), int(float(bb[1])+float(bb[3])))

                        bound1 = (max(0, d[0]), max(0, d[1]),
                                  min(d[2], width), min(d[3], height))
                        img_temp = img_cv[bound1[1]:bound1[3], bound1[0]:bound1[2]]
                        
                        img_temp_cls = cv2.cvtColor(img_temp, cv2.COLOR_RGB2BGR)
                        start = time.time()
                        # print(bound1)
                        # run classification
                        img_temp_cls = cv2.resize(img_temp_cls, (299, 299), cv2.INTER_LINEAR)
                        img_temp_cls = np.array(img_temp_cls, dtype = np.uint8)
                        img_temp_cls = img_temp_cls.astype('float32')
                        img_temp_cls = np.multiply(img_temp_cls, 1.0/255.0)
                        img_temp_cls = np.subtract(img_temp_cls, 0.5)
                        img_temp_cls = np.multiply(img_temp_cls, 2.0)
                        img_temp_cls = np.expand_dims(img_temp_cls, 0)
                            # images = list()
                            # # crop from img_tf, create a new image img_tf2, then process , convert to array ,and run 
                            # img_tf2 = tf.image.crop_to_bounding_box(img_tf, bound1[1], bound1[0], bound1[3]-bound1[1], bound1[2]-bound1[0])
                            # # plt.imshow(img_tf2.eval())
                            # # plt.show()
                            # processed_image = image_preprocessing_fn(img_tf2, test_image_size, test_image_size)
                            # processed_image = sess.run(processed_image)
                            # images.append(processed_image)
                            # images = np.array(images)
                        predictions = sess.run(logits, feed_dict = {tensor_input : img_temp_cls}).indices
                        new_bbs_txt.write('{} {} {} {} {} {}\n'.format(bb[4], bb[0], bb[1], bb[2], bb[3], predictions[0]))

                        if predictions[0] == 0:
                            cv2.imwrite(save_root + 'fro/' + img_name[:-4] + '_' + bb[4] + '.jpg', img_temp)
                        else:
                            cv2.imwrite(save_root + 'pro/' + img_name[:-4] + '_' + bb[4] + '.jpg', img_temp)

                        end = time.time()
                        print( '{:.2} seconds per bnx\n'.format(end - start))
                        # 1 : profile 0: frontal
                    new_bbs_txt.close()
                    gc.collect()
                        # if predictions[0] == 1:
                        #     # print( type(save_root), type(img_name[:-4]), type(img_temp))
                        # else:

if __name__ == '__main__':
    tf.app.run()


#### compute precision by raw_split directory (split profile/ frontal by myself)
# for bb in bbs:
# 	if bb[5] == 'DXP':
# 		d = (int(float(bb[0])), int(float(bb[1])),
#                   int(float(bb[0]) + float(bb[2])), int(float(bb[1])+float(bb[3])))
#         bound1 = (max(0, d[0]), max(0, d[1]),
#                   min(d[2], width), min(d[3], height))
#         img_temp = img_cv[bound1[1]:bound1[3], bound1[0]:bound1[2]]

#         # print(bound1)
#         # run classification
        
#         images = list()
#         # crop from img_tf, create a new image img_tf2, then process , convert to array ,and run 
#         img_tf2 = tf.image.crop_to_bounding_box(img_tf, bound1[1], bound1[0], bound1[3]-bound1[1], bound1[2]-bound1[0])
#         # plt.imshow(img_tf2.eval())
#         # plt.show()
#         processed_image = image_preprocessing_fn(img_tf2, test_image_size, test_image_size)
#         processed_image = sess.run(processed_image)
#         images.append(processed_image)
#         images = np.array(images)
#         predictions = sess.run(logits, feed_dict = {tensor_input : images}).indices

#         if os.path.exists('raw_split/DXP/middle/' + img_name) or os.path.exists('raw_split/DXP/youth/' + img_name):
#         	break

#         if os.path.exists('raw_split/DXP/' + img_name):
#         	gt = 0 # frontal
#         else:
#         	gt = 1

#         if gt == predictions[0]:
#         	right += 1
#         else:
#         	wrong += 1
# print(right , wrong)