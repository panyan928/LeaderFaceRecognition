from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,json,math,time,glob,cv2
import numpy as np
import tensorflow as tf
from nets import nets_factory
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
    save_root = './detect_compute'
    label_path_root = './result2'
    bbs_path_root = './txts'
    img_path_root = './raw'
    def BaseFeat(self, leader):
        feature_fro = list()
        feature_pro = list()
        bases = glob.glob('raw3/' + leader + '-base/' +leader+'_*.jpg')
        # bases = glob.glob('raw2/' + leader + '-base/*.jpg')
        for n, path in enumerate(bases):
            label = os.path.basename(path)[:-4] + ';0;'

            # print(label)

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
            f, _  = model.get_feature_by_landmark(img, bound1, preds)

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
            if bbs[0][6] == 1:
                feature_pro.append(f)
            elif bbs[0][6] == 0:
                feature_fro.append(f)
            else:
                print('no profile / frontal file\n')
        return feature_fro, feature_pro



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
        parser = argparse.ArgumentParser(description='face model test')
        parser.add_argument('--image-size', default='112,112', help='')
        parser.add_argument('--model', default='./model-r50-am-lfw/model,0', help='path to load model.')
        parser.add_argument('--gpu', default=0, type=int, help='gpu id')
        parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
        parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        args = parser.parse_args()
        model = face_embedding.FaceModel(args)

        ###############################
        # Deploy file path and params #
        ###############################
        cfg = Config()
        feat_box_fro = dict()
        feat_box_pro = dict()
        output = open('result_txt/result_align_sin_2_all.txt', 'w')
        output.write('total leaders: {}'.format(len(cfg.leaders_sml[:-1])))
        countDet = 0
        countData = [0, 0]
        no_label = 0
        right = [0, 0]
        FN = 0
        FP = 0
        num = 0

        true_profile = 0
        true_frontal = 0
        false_profile = 0
        false_frontal = 0

        for leader in cfg.leaders_sml[:-1]:

            save_root = '{}/{}/'.format(cfg.save_root, leader)
            f1_fro, f1_pro = cfg.BaseFeat(leader)
            feat_box_fro[leader] = f1_fro
            print(leader)
            feat_box_pro[leader] = f1_pro
            print('frontal: {} profile: {}'.format(len(f1_fro), len(f1_pro)))
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
            saver.restore(sess, checkpoint_path)

            ###############################
            # Detect image Loop by leader #
            ###############################
            for leader in cfg.leaders_sml[:-1]:
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
                    ### there need change!! PLY_01700 have no number but have others groundturth
                    if len(label) == 10:
                        label = label[:-1]
                    print(label)
                    img_name = label.split(';')[0]+'.jpg'
                    # load image
                    img_cv = cv2.imread(img_path + img_name)
                    if img_cv is None:
                        img_name = img_name[:-4]+'.JPG'
                        img_cv = cv2.imread(img_path + img_name)
                    # img_tf = open(img_path + img_name, 'rb').read()
                    # img_tf = tf.image.decode_jpeg(img_tf, channels=3)

                    # Get groundtruth of one image 
                    # bbs: '[x, y, w, h, num, leader]'
                    gt = utils.get_bbox(bbs_path, label, default_leader = leader)
                    if gt is None:
                        continue
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

                    for i, d in enumerate(bbox):
                        # bound1 = (float(d[0]), float(d[1]), float(d[2]), float(d[3]))
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
                        bound1 = (max(0, int(d[0])), max(0, int(d[1])),
                                  min(int(d[2]), width), min(int(d[3]), height))

                        if iou < 0.5:
                            print("the face no label")
                            cv2.imwrite('{}no_label/{}_{}.jpg'.format(save_root, img_name[:-4], i), img_cv[bound1[1]:bound1[3], bound1[0]:bound1[2]])
                            # cv2.putText(img_cv, str(i) + 'no label', (bound1[0], bound1[1]), 0, 1, (255, 0, 0))
                            # cv2.imwrite(save_root +'no_label/' +img_name + '.jpg', img_cv)
                            no_label += 1
                            continue


                        ## face landmark detection
                        preds = fa.get_landmarks(img_path + img_name, bound1)[-1]
                        ## face embedding 512d
                        f, img_aligned = model.get_feature_by_landmark(img_cv, bound1, preds)

                        # d = [int(round(d[0])), int(round(d[1])), int(round(d[2])), int(round(d[3]))]
                        
                        ### Add a step
                        ### classify the face profile or frontal
                        ### groundtruth no pose information, classify by netwrok
                        if pose == -1:
                            print('Run Classification Model')
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

                        if pose == 1:
                            threshold = 0.85
                           
                        if pose == 0:
                            threshold = 1.1
                            
                        countDet += 1

                        score = 999
                        result = 'OTHERS'
                        for feat_leader in cfg.leaders_sml[:-1]:
                            print(feat_leader)
                            for f2 in (feat_box_pro[feat_leader] + feat_box_fro[feat_leader]):
                                dist = np.sum(np.square(f - f2))
                                
                                if dist < threshold and dist < score:
                                    score = dist
                                    result = feat_leader

                        if result == positive: 
                            if pose == 0:
                                true_frontal += 1
                            if pose == 1:
                                true_profile += 1

                            if result  == 'OTHERS':
                                right[1] += 1
                                cv2.imwrite('{}right1/{}_{}.jpg'.format(save_root, img_name[:-4], i), img_aligned)

                            else:
                                right[0] += 1
                                cv2.imwrite('{}right0/{}_{}.jpg'.format(save_root, img_name[:-4], i), img_aligned)
                        else:
                            if pose == 0:
                                false_frontal += 1
                            if pose == 1:
                                false_profile += 1
                            if positive != 'OTHERS':
                                cv2.imwrite('{}falsePositive/{}_{}_{}.jpg'.format(save_root, img_name[:-4], i, result), img_aligned)
                                FP += 1
                            else:
                                cv2.imwrite('{}falseNegative/{}_{}_{}.jpg'.format(save_root, img_name[:-4], i, result), img_aligned)
                                FN += 1

        
                output.write('countData:{} leaders,{} others \n Sum:{} countDet:{} no_label:{}\n'.format( countData[0],countData[1], num, countDet, no_label )) 
                output.write('detection recall:{}\n'.format(countDet/float(countData[0]+countData[1])))

                precision = float(right[0]+right[1]) / (countDet)
                recall_leader = float(right[0]) / countData[0]
                recall_all = float(right[0]+right[1]) / (countData[0]+countData[1])
                
                profile_precision = true_profile / (true_profile + false_profile)
                frontal_precision = true_frontal / (true_frontal + false_frontal)

                output.write('right:{},{} falsePositive:{} falseNegative:{}\n'.format(right[0], right[1], FP, FN))
                output.write('match precision :{}\n'.format(precision))
                output.write('leader match recall:{} all recall:{}\n'.format(recall_leader, recall_all))
                output.write('true profile:{} false profile:{} profile precision:{}\n'.format(true_profile, false_profile, profile_precision))
                output.write('true frontal:{} false frontal:{} frontal precision:{}\n'.format(true_frontal, false_frontal, frontal_precision))
