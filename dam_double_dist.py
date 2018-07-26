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
import face_align

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
    #leaders_sml = ['DXP']
    save_root = './detect_compute'
    label_path_root = './result2'
    bbs_path_root = './txts'
    img_path_root = './raw'




    def BaseFeat(self, leader):
        feature = list()
        bases = glob.glob('raw3/' + leader + '-base/' +leader+'_*.jpg')
        # bases = glob.glob('raw2/' + leader + '-base/*.jpg')
        for n, path in enumerate(bases):
            label = os.path.basename(path)[:-4] + ';0;'
            print(label)
            # if label[7:9] =='61':
            #     img = cv2.imread(path)
            #     img = img[:, int(img.shape[1]/2):img.shape[1]]
            #     boxes,points = model.detect(img)
            #     f = model.get_feature(img, boxes[0], points[0])
            #     img = img[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]
            #     cv2.imwrite('DXP_00061_2.jpg', img)
            #     if f is None: print('DXP_00061.jpg is None')
            #     feature.append(f)
            #     continue

            img = cv2.imread(path)
            # print(img.shape)
            bbs, bbs_other = utils.get_bbox('txts/' + leader + '/', label, default_leader=leader)
            bound1 = (float(bbs[0][0]), float(bbs[0][1]), float(bbs[0][2])+float(bbs[0][0]), float(bbs[0][3])+float(bbs[0][1]))
            preds = fa.get_landmarks(path , bound1)[-1]
                        ## face embedding 512d

            #xmq

            #f = model.get_feature_by_landmark(img, bound1, preds)

            f_1 = model1.get_feature_by_landmark(img, bound1, preds)
            f_2 = model2.get_feature_by_landmark(img, bound1, preds)

            f = (f_1,f_2)

            # for i in range(0, int(max(float(bbs[0][2]), float(bbs[0][3])))):
            #     bbs_temp = (max(0, int(float(bbs[0][0])) - i), 
            #                 max(0, int(float(bbs[0][1])) - i),
            #                 min(img.shape[1], int(float(bbs[0][0]) + float(bbs[0][2]) + i)),
            #                 min(img.shape[0], int(float(bbs[0][1]) + float(bbs[0][3]) + i)))
            #     img_temp = img[bbs_temp[1]:bbs_temp[3], bbs_temp[0]:bbs_temp[2]]
            #     # cv2.imshow("crop", img_temp)
            #     # cv2.waitKey(0)
            #     f = model.get_feature_limited(img_temp)
            #     if f is not None:
            #         # cv2.imwrite('raw2/' + leader + '-base/' +str(n)+'.jpg', img_temp)
            #         break
            # if f is None: print(path + ' is None')
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

        #################################
        # Load face detector model RFCN #
        #################################
        prototxt = '/home/disk3/py/py-R-FCN/models/pascal_voc/ResNet-101/rfcn_end2end/test_agonistic_face.prototxt'
        caffemodel = '/home/disk3/py/py-R-FCN/data/rfcn_models/resnet101_rfcn_ohem_iter_40000.caffemodel'
        detect_cfg.TEST.HAS_RPN = True
        caffe.set_mode_gpu()
        caffe.set_device(0)
        detect_cfg.GPU_ID = 0
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        print ('\n\nLoaded network {:s}'.format(caffemodel))

        ##################################
        # Load landmark detect and align #
        ##################################
        fa = face_align.FaceAlignment(face_align.LandmarksType._2D, enable_cuda=True)

        ####################
        # Load InsightFace #
        ####################

        #xmq

        #load model1

        parser = argparse.ArgumentParser(description='face model test')
        parser.add_argument('--image-size', default='112,112', help='')
        parser.add_argument('--model', default='./model-r50-am-lfw/model,0', help='path to load model.')
        parser.add_argument('--gpu', default=0, type=int, help='gpu id')
        parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
        parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        args = parser.parse_args()
        model1 = face_embedding.FaceModel(args)


        #load model2

        parser = argparse.ArgumentParser(description='face model test')
        parser.add_argument('--image-size', default='112,112', help='')
        parser.add_argument('--model', default='./model/model,2000', help='path to load model.')
        parser.add_argument('--gpu', default=0, type=int, help='gpu id')
        parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
        parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        args = parser.parse_args()
        model2 = face_embedding.FaceModel(args)







        ###############################
        # Deploy file path and params #
        ###############################
        cfg = Config()
        feat_box = dict()
        output = open('result_txt/result_xmq.txt', 'w')
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

        true_profile = dict() 
        true_frontal = dict()
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

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)

            ###############################
            # Detect image Loop by leader #
            ###############################
            for leader in ['DXP','JZM', 'MZD', 'PLY', 'XJP']:
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
                    if len(label) <= 10:
                        continue
                    print(label)
                    img_name = label.split(';')[0]+'.jpg'
                    # load image
                    img_cv = cv2.imread(img_path + img_name)
                    if img_cv is None:
                        img_name = img_name[:-4]+'.JPG'
                        img_cv = cv2.imread(img_path + img_name)
                    img_tf = open(img_path + img_name, 'rb').read()
                    img_tf = tf.image.decode_jpeg(img_tf, channels=3)
                    ###############################
                    # Detect all faces in a image #
                    ###############################
                    scores, boxes = im_detect(net, img_cv)
                    height, width, c = img_cv.shape
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

                    # Get groundtruth of one image 
                    # bbs: '[x, y, w, h, num, leader]'
                    bbs, bbs_other = utils.get_bbox(bbs_path, label, default_leader = leader)
                    countData[0] += len(bbs) # 
                    countData[1] += len(bbs_other)
                    # new_bbs_txt = open(bbs_path + img_name[:-4] + '_new.txt', 'w')

                    for d in bbox:
                        bound1 = (float(d[0]), float(d[1]), float(d[2]), float(d[3]))
                        ## face landmark detection
                        preds = fa.get_landmarks(img_path + img_name, bound1)[-1]
                        ## face embedding 512d

                        #xmq

                        #f = model.get_feature_by_landmark(img_cv, bound1, preds)
                        f_1 = model1.get_feature_by_landmark(img_cv, bound1, preds)
                        f_2 = model2.get_feature_by_landmark(img_cv, bound1, preds)

                        #print(f_1)

                        f = (f_1,f_2)

                        # d = [int(round(d[0])), int(round(d[1])), int(round(d[2])), int(round(d[3]))]
                        bound1 = (max(0, bound1[0]), max(0, bound1[1]),
                                  min(bound1[2], width), min(bound1[3], height))

                        countDet += 1
                        for s in score:
                            score[s] = 999
                            result[s] = 'OTHERS'
                        for feat_leader in feat_box:
                            for f2 in feat_box[feat_leader]:

                                #xmq
                                #print(f[0])
                                #print(f2[0])

                                dist1 = np.sum(np.square(f[0][0] - f2[0][0]))


                                dist2 = np.sum(np.square(f[1][0] - f2[1][0]))

                                #print("dist1:%f"%dist1)
                                #print("dist2:%f"%dist2)

                                for threshold in range(60, 130, 5):
                                    threshold = threshold * 0.01
                                    if dist1 < threshold and dist1 < score[threshold]:
                                        score[threshold] = dist1
                                        result[threshold] = feat_leader
                                    if dist2 < threshold and dist2 < score[threshold]:
                                        score[threshold] = dist2
                                        result[threshold] = feat_leader

                        iou = 0
                        positive = 'OTHERS'
                        pose = -1
                        for bb in (bbs + bbs_other):
                            bound2 = (float(bb[0]), float(bb[1]),
                                      float(bb[0]) + float(bb[2]), float(bb[1])+float(bb[3]))
                            iou = calculateIoU(bound1, bound2)
                            if iou >= 0.5:
                                positive = bb[5]
                                pose = bb[6]
                                break
                        if iou < 0.5:
                            print("the face no label")
                            # cv2.putText(img_cv, str(i) + 'no label', (bound1[0], bound1[1]), 0, 1, (255, 0, 0))
                            # cv2.imwrite(save_root +'no_label/' +img_name + '.jpg', img_cv)
                            no_label += 1
                            continue
                        for threshold in range(60, 130, 5):
                            threshold = threshold * 0.01
                            if result[threshold] == positive: 
                                if result[threshold] == 'OTHERS':
                                    right[threshold][1] += 1
                                    if pose == 0:
                                        true_frontal[threshold] += 1
                                    elif pose == 1:
                                        true_profile[threshold] += 1
                                else:
                                    right[threshold][0] += 1
                                    if pose == 0:
                                        true_frontal[threshold] += 1
                                    elif pose == 1:
                                        true_profile[threshold] += 1
                            elif positive != 'OTHERS': 
                                # cv2.putText(img_cv, str(i) + result[threshold] + ' ' + positive, (bound1[0], bound1[1]), 0, 0.5, (0, 255, 0))
                                # cv2.imwrite(save_root + 'falsePositive/' + img_name + '.jpg', img_cv)
                                FP[threshold] += 1
                                if pose == 0:
                                    false_frontal[threshold] += 1
                                elif pose == 1:
                                    false_profile[threshold] += 1
                            else: # result != 'OTHERS': 
                                FN[threshold] += 1
                                if pose == 0:
                                    false_frontal[threshold] += 1
                                elif pose == 1:
                                    false_profile[threshold] += 1

                        # img_temp = img_cv[bound1[1]:bound1[3], bound1[0]:bound1[2]]
                        # print(bound1)
                        # run classification
                        
                    #     images = list()
                    #     # crop from img_tf, create a new image img_tf2, then process , convert to array ,and run 
                    #     img_tf2 = tf.image.crop_to_bounding_box(img_tf, bound1[1], bound1[0], bound1[3]-bound1[1], bound1[2]-bound1[0])
                    #     # plt.imshow(img_tf2.eval())
                    #     # plt.show()
                    #     processed_image = image_preprocessing_fn(img_tf2, test_image_size, test_image_size)
                    #     processed_image = sess.run(processed_image)
                    #     images.append(processed_image)
                    #     images = np.array(images)
                    #     predictions = sess.run(logits, feed_dict = {tensor_input : images}).indices 
                    #     order = -1
                    #     # find groundtruth by IOU
                    #     for bb in (bbs + bbs_other):
                    #         bound2 = (float(bb[0]), float(bb[1]),
                    #                   float(bb[0]) + float(bb[2]), float(bb[1])+float(bb[3]))
                    #         # print(bound1, bound2)
                    #         iou = calculateIoU(bound1, bound2)
                    #         if iou >= 0.5:
                    #             bound1 = bb
                    #             order = int(bb[4])
                    #             break
                    #     if order != -1:
                    #         new_bbs_txt.write('{} {} {} {} {} {}\n'.format(bound1[4], bound1[0], bound1[1], bound1[2], bound1[3], predictions[0]))
                    #         if predictions[0] == 0:
                    #             cv2.imwrite(save_root + 'fro/' + img_name[:-4] + '_' + bound1[4] + '.jpg', img_temp)
                    #         else:
                    #             cv2.imwrite(save_root + 'pro/' + img_name[:-4] + '_' + bound1[4] + '.jpg', img_temp)
                    #     # 1 : profile 0: frontal
                    # new_bbs_txt.close()

                        # if predictions[0] == 1:
                        #     # print( type(save_root), type(img_name[:-4]), type(img_temp))
                        # else:

                    for threshold in range(60, 130, 5):
                        threshold = threshold * 0.01
                        precision[threshold] = float(right[threshold][0] + right[threshold][1]) / (countDet - no_label)
                        recall[threshold] = float(right[threshold][0]) / countData[0]
                        if true_profile[threshold] + false_profile[threshold] == 0:
                            profile_precision = 0
                        else:
                            profile_precision = true_profile[threshold] / (true_profile[threshold] + false_profile[threshold])
                        if true_frontal[threshold] + false_frontal[threshold] == 0:
                            frontal_precision = 0
                        else:
                            frontal_precision = true_frontal[threshold] / (true_frontal[threshold] + false_frontal[threshold])
                        '''
                        output.write('threshold:{}\n'.format(threshold))
                        output.write('right:{},{} falsePositive:{} falseNegative:{}\n'.format(right[threshold][0], 
                            right[threshold][1], FP[threshold], FN[threshold]))
                        output.write('match precision :{}\n'.format(precision[threshold]))
                        output.write('leader match recall:{}\n'.format(recall[threshold]))
                        output.write('true profile:{} false profile:{} profile precision:{}\n'.format(true_profile[threshold], false_profile[threshold], 
                            profile_precision))
                        output.write('true frontal:{} false frontal:{} frontal precision:{}\n'.format(true_frontal[threshold], false_frontal[threshold], 
                            frontal_precision))
                        '''
        
                output.write('countData:{} leaders,{} others \n Sum:{} countDet:{} no_label:{}\n'.format( countData[0],countData[1], num, countDet, no_label )) 
                output.write('detection recall:{}\n'.format(num/float(countData[0]+countData[1])))
                for threshold in range(60, 130, 5):
                    threshold = threshold * 0.01
                    precision[threshold] = float(right[threshold][0]+right[threshold][1])/(countDet - no_label)
                    recall[threshold] = float(right[threshold][0]) / countData[0]
                    output.write('threshold:{}\n'.format(threshold))
                    output.write('right:{},{} falsePositive:{} falseNegative:{}\n'.format(right[threshold][0], 
                        right[threshold][1], FP[threshold], FN[threshold]))
                    output.write('match precision :{}\n'.format(precision[threshold]))
                    output.write('leader match recall:{}\n'.format(recall[threshold]))
                    output.write('true profile:{} false profile:{} profile precision:{}\n'.format(true_profile[threshold], false_profile[threshold], 
                        true_profile[threshold]/(true_profile[threshold]+false_profile[threshold])))
                    output.write('true frontal:{} false frontal:{} frontal precision:{}\n'.format(true_frontal[threshold], false_frontal[threshold], 
                        true_frontal[threshold]/(true_frontal[threshold]+false_frontal[threshold])))
