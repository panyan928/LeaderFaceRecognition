from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,json,math,time,glob,cv2,shutil
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
def less_average(score):
  num = len(score)
  sum_score = sum(score)
  ave_num = sum_score/num
  # less_ave = [i for i in score if i<ave_num]
  return ave_num

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
      
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)

            ###############################
            # Detect image Loop by leader #
            ###############################
            cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
            dirs = os.listdir('vgg2face_test')
            for subdir in dirs:
                if subdir[0] != 'n':
                    continue
                img_paths = os.listdir('vgg2face_test/' + subdir)

                for img_path in img_paths:
                    print(subdir + '/' +img_path)
                    # if os.path.exists('vgg2face_test/frontal/'+ subdir + '/' + img_path):
                    #     continue
                    # if os.path.exists('vgg2face_test/profile/'+ subdir + '/' + img_path):
                    #     continue
                    img = cv2.imread('vgg2face_test/' + subdir + '/' +img_path)
                    # img_draw = img
                    height, width, _ = img.shape

                    scores, boxes = im_detect(net, img)
                    thresh = 0.95
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

                    for i, d in enumerate(bbox):
                        is_complete = True
                        ## face landmark detection
                        preds = fa.get_landmarks('vgg2face_test/' + subdir + '/' + img_path, d)[-1]

                        ## draw landmarks
                        lefteye_x =int(less_average([preds[36,0],preds[37,0],preds[38,0],preds[39,0],preds[40,0],preds[41,0]]))
                        lefteye_y =int(less_average([preds[36,1],preds[37,1],preds[38,1],preds[39,1],preds[40,1],preds[41,1]]))

                        righteye_x =int(less_average([preds[42,0],preds[43,0],preds[44,0],preds[45,0],preds[46,0],preds[47,0]]))
                        righteye_y =int(less_average([preds[42,1],preds[43,1],preds[44,1],preds[45,1],preds[46,1],preds[47,1]]))

                        nose_x =int(less_average([preds[27,0],preds[28,0],preds[29,0],preds[30,0],preds[32,0],preds[33,0],preds[34,0]]))
                        nose_y =int(less_average([preds[27,1],preds[28,1],preds[29,1],preds[30,1],preds[32,1],preds[33,1],preds[34,1]]))

                        leftmouth_x =int(less_average([preds[48,0],preds[49,0],preds[60,0],preds[59,0]]))
                        leftmouth_y =int(less_average([preds[48,1],preds[49,1],preds[60,1],preds[59,1]]))

                        rightmouth_x =int(less_average([preds[54,0],preds[53,0],preds[64,0],preds[55,0]]))
                        rightmouth_y =int(less_average([preds[54,1],preds[53,1],preds[64,1],preds[55,1]]))
                        
                        five = []
                        five.append(lefteye_x)
                        five.append(righteye_x)
                        five.append(nose_x)
                        five.append(leftmouth_x)
                        five.append(rightmouth_x)
                        five.append(lefteye_y)
                        five.append(righteye_y)            
                        five.append(nose_y)            
                        five.append(leftmouth_y)            
                        five.append(rightmouth_y)
                        for j in range(0,5):
                            if five[j] < 0 or five[j+5] <0 :
                                is_complete = False
                                print(five[j], five[j+5])
                            ## landmark beyond images, isn't a full face , pass it 
                            if five[j] >= width or five[j+5] >= height:
                                is_complete = False
                            # cv2.putText(img_draw, str(j), (five[j]-1, five[j+5]-1), 0, 0.5, (0,0,255))
                            # cv2.circle(img_draw, (five[j], five[j+5]), 1, (0, 255, 255), 2)
                        if is_complete is False:
                            continue

                        ## face embedding 512d
                        f, nimg = model.get_feature_by_landmark(img, d, preds)
                        img_temp_cls = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
                        img_temp_cls = cv2.resize(img_temp_cls, (299, 299), cv2.INTER_LINEAR)
                        img_temp_cls = np.array(img_temp_cls, dtype = np.uint8)
                        img_temp_cls = img_temp_cls.astype('float32')
                        img_temp_cls = np.multiply(img_temp_cls, 1.0/255.0)
                        img_temp_cls = np.subtract(img_temp_cls, 0.5)
                        img_temp_cls = np.multiply(img_temp_cls, 2.0)
                        img_temp_cls = np.expand_dims(img_temp_cls, 0)
                        predictions = sess.run(logits, feed_dict = {tensor_input : img_temp_cls}).indices
                        pose = predictions[0]
                        # if os.path.exists('vgg2face_test/frontal/' + subdir) is False:
                        #   os.mkdir('vgg2face_test/frontal/' + subdir)
                        # if not os.path.exists('vgg2face_test/profile/' + subdir):
                        #   os.mkdir('vgg2face_test/profile/' + subdir)
                        if pose == 0:
                          cv2.imwrite('vgg2face_test/frontal/' + subdir + '_' + img_path[:-4] +' _'+ str(i) + '.jpg', nimg)
                        else:
                          cv2.imwrite('vgg2face_test/profile/' + subdir + '_' + img_path[:-4] +' _'+ str(i) + '.jpg', nimg)


