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
    image =cv2.imread(img_path)
    print(bb)
    #bb = bb[0]
    bbs = []
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
    image = image[bbs[1]:bbs[1]+bbs[3],bbs[0]:bbs[0]+bbs[2]]
    cv2.imwrite('pic_store/exchange/1.jpg',image)
    return image









tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/disk3/py/tensorflow/models-master/models-master/research/slim/train_vgg_logs/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

#tf.app.flags.DEFINE_string(
#    'test_path', 'fpface/frontal/f-0000000.jpg', 'Test image path.')

tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'test_image_size', 112, 'Eval image size')

FLAGS = tf.app.flags.FLAGS





txt_path = 'result2/'

bbs_path = 'txts/'

pic_path = 'raw/'

exchange_path = 'pic_store/exchange/'


pic_save_path = 'pic_store/fp/'

dirs = os.listdir(txt_path)

line_num = 0

learder_name_list = ['dxp','jzm','mzd','ply','xjp']

dict_learder_num = {'dxp':0,'jzm':0,'mzd':0,'ply':0,'xjp':0}







def main(_):
    #if not FLAGS.test_list:
    #    raise ValueError('You must supply the test list with --test_list')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
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

        tf.Graph().as_default()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #saver = tf.train.Saver()
            #saver.restore(sess, checkpoint_path)

            #image = open('pic_store/exchange/1.jpg', 'rb').read()

            #image = image[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
            #image = tf.image.decode_jpeg(image, channels=3)

            #processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)

            #processed_images = tf.expand_dims(processed_image, 0)
            #logits, _ = network_fn(processed_images)
            #predictions = tf.argmax(logits, 1)
            #saver = tf.train.Saver()
            #saver.restore(sess, checkpoint_path)
            #np_image, network_input, predictions = sess.run([image, processed_image, predictions]



            ####################leader_label_test########################

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

                    label_split = line.split(';') 

                    pic_name = label_split[0]

                    pic_bb_txt_path = bbs_path + subdir.upper() + '/' + pic_name + '_new' + '.txt'

                    pic_bb_txt = open(pic_bb_txt_path,'r')

                    pic_bb_txt_lines = pic_bb_txt.readlines()

                    pic_path_full = pic_path_f + pic_name + '.jpg'



                    l = len(label_split)
                    #print(len(label_split))

                    #if l ==2:
                    #   print(l)

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

                                image_ex = open('pic_store/exchange/1.jpg', 'rb').read()

                                image_ex = tf.image.decode_jpeg(image_ex, channels=3)

                                processed_image = image_preprocessing_fn(image_ex, test_image_size, test_image_size)

                                processed_images = tf.expand_dims(processed_image, 0)
                                logits, _ = network_fn(processed_images)
                                predictions = tf.argmax(logits, 1)

                                np_image, network_input, pose = sess.run([image_ex, processed_image, predictions])

                                #pose = get_fp()
                                print(pose)

                                save_p = pic_save_path + learder_name + '/' + str(pose) + '/' + pic_name + '_' + other + '.jpg' 

                                cv2.imwrite(save_p,image)

                            else:

                                other_no = label_split[j].split(':')[0]
                                other_name = label_split[j].split(':')[1]
                                if other_name not in learder_name_list:
                                    print('other leader')
                                    bb = get_bb(pic_bb_txt_lines,other_no)


                                    image=crop_pic(pic_path_full,bb)

                                    image_ex = open('pic_store/exchange/1.jpg', 'rb').read()

                                    image_ex = tf.image.decode_jpeg(image_ex, channels=3)

                                    processed_image = image_preprocessing_fn(image_ex, test_image_size, test_image_size)

                                    processed_images = tf.expand_dims(processed_image, 0)

                                    logits, _ = network_fn(processed_images)
                                    predictions = tf.argmax(logits, 1)

                                    np_image, network_input, pose = sess.run([image_ex, processed_image, predictions])

                                    #pose = get_fp()
                                    print(pose)

                                    save_p = pic_save_path + 'other' + '/' + str(pose) + '/' + pic_name + '_' + other_no + '.jpg' 

                                    cv2.imwrite(save_p,image)

                                else:
                                    dict_learder_num[other_name] +=1
                                    print('learder %s no %s'%(other_name,other_no))
                                    bb = get_bb(pic_bb_txt_lines,other_no)

                                    image=crop_pic(pic_path_full,bb)

                                    image_ex = open('pic_store/exchange/1.jpg', 'rb').read()

                                    image_ex = tf.image.decode_jpeg(image_ex, channels=3)

                                    processed_image = image_preprocessing_fn(image_ex, test_image_size, test_image_size)

                                    processed_images = tf.expand_dims(processed_image, 0)

                                    logits, _ = network_fn(processed_images)
                                    predictions = tf.argmax(logits, 1)

                                    np_image, network_input, pose = sess.run([image_ex, processed_image, predictions])

                                    #pose = get_fp()
                                    print(pose)

                                    save_p = pic_save_path + other_name + '/' + str(pose) + '/' + pic_name + '_' + other_no + '.jpg' 

                                    cv2.imwrite(save_p,image)


                    elif l ==3 and int(label_split[1]) == -1:
                        print('no learder')
                    elif l ==3 and int(label_split[1]) > -1:
                        print('learder %s no %s'%(learder_name,label_split[1]))
                        dict_learder_num[learder_name] +=1
                        bb = get_bb(pic_bb_txt_lines,label_split[1])

                        image=crop_pic(pic_path_full,bb)

                        #image = open(FLAGS.test_path, 'rb').read()
            #image = tf.image.decode_jpeg(image, channels=3)

            #processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)

            #processed_images = tf.expand_dims(processed_image, 0)
            #logits, _ = network_fn(processed_images)
            #predictions = tf.argmax(logits, 1)
            #saver = tf.train.Saver()
            #saver.restore(sess, checkpoint_path)
            #np_image, network_input, predictions = sess.run([image, processed_image, predictions])
            #print('{} {}'.format(FLAGS.test_path, predictions[0]))

                        #processed_image = image_preprocessing_fn(img_tf2, test_image_size, test_image_size)
                        #processed_image = sess.run(processed_image)
                    #     images.append(processed_image)
                    #     images = np.array(images)
                    #     predictions = sess.run(logits, feed_dict = {tensor_input : images}).indices 

                        image_ex = open('pic_store/exchange/1.jpg', 'rb').read()

                        image_ex = tf.image.decode_jpeg(image_ex, channels=3)

                        processed_image = image_preprocessing_fn(image_ex, test_image_size, test_image_size)

                        processed_images = tf.expand_dims(processed_image, 0)

                        logits, _ = network_fn(processed_images)
                        predictions = tf.argmax(logits, 1)

                        saver = tf.train.Saver()
                        saver.restore(sess, checkpoint_path)

                        np_image, network_input, pose = sess.run([image_ex, processed_image, predictions])

                        #pose = get_fp()
                        print(pose)

                        save_p = pic_save_path + learder_name + '/' + str(pose) + '/' + pic_name + '_' + label_split[1] + '.jpg' 

                        cv2.imwrite(save_p,image)
                    elif l >3 and len(label_split[1].split(':'))==1:

                        if int(label_split[1]) == -1:
                            print('no leader')
                        elif int(label_split[1]) > -1:
                            print('learder %s no %s'%(learder_name,label_split[1]))
                            dict_learder_num[learder_name] +=1
                            bb = get_bb(pic_bb_txt_lines,label_split[1])


                            image=crop_pic(pic_path_full,bb)

                            image_ex = open('pic_store/exchange/1.jpg', 'rb').read()

                            image_ex = tf.image.decode_jpeg(image_ex, channels=3)

                            processed_image = image_preprocessing_fn(image_ex, test_image_size, test_image_size)

                            processed_images = tf.expand_dims(processed_image, 0)

                            logits, _ = network_fn(processed_images)
                            predictions = tf.argmax(logits, 1)

                            np_image, network_input, pose = sess.run([image_ex, processed_image, predictions])

                            #pose = get_fp()
                            print(pose)

                            save_p = pic_save_path + learder_name + '/' + str(pose) + '/' + pic_name + '_' + label_split[1] + '.jpg' 

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

                                image_ex = open('pic_store/exchange/1.jpg', 'rb').read()

                                image_ex = tf.image.decode_jpeg(image_ex, channels=3)

                                processed_image = image_preprocessing_fn(image_ex, test_image_size, test_image_size)

                                processed_images = tf.expand_dims(processed_image, 0)

                                logits, _ = network_fn(processed_images)
                                predictions = tf.argmax(logits, 1)

                                np_image, network_input, pose = sess.run([image_ex, processed_image, predictions])

                                #pose = get_fp()
                                print(pose)

                                save_p = pic_save_path + learder_name + '/' + str(pose) + '/' + pic_name + '_' + other + '.jpg' 

                                cv2.imwrite(save_p,image)
                            else:

                                other_no = label_split[j].split(':')[0]
                                other_name = label_split[j].split(':')[1]
                                if other_name not in learder_name_list:
                                    print('other leader')
                                    bb = get_bb(pic_bb_txt_lines,other_no)

                                    image=crop_pic(pic_path_full,bb)

                                    image_ex = open('pic_store/exchange/1.jpg', 'rb').read()

                                    image_ex = tf.image.decode_jpeg(image_ex, channels=3)

                                    processed_image = image_preprocessing_fn(image_ex, test_image_size, test_image_size)

                                    processed_images = tf.expand_dims(processed_image, 0)

                                    logits, _ = network_fn(processed_images)
                                    predictions = tf.argmax(logits, 1)

                                    np_image, network_input, pose = sess.run([image_ex, processed_image, predictions])

                                    #pose = get_fp()
                                    print(pose)

                                    save_p = pic_save_path + 'other' + '/' + str(pose) + '/' + pic_name + '_' + other_no + '.jpg' 

                                    cv2.imwrite(save_p,image)
                                else:
                                    dict_learder_num[other_name] +=1
                                    print('learder %s no %s'%(other_name,other_no))
                                    bb = get_bb(pic_bb_txt_lines,other_no)

                                    image=crop_pic(pic_path_full,bb)

                                    image_ex = open('pic_store/exchange/1.jpg', 'rb').read()

                                    image_ex = tf.image.decode_jpeg(image_ex, channels=3)

                                    processed_image = image_preprocessing_fn(image_ex, test_image_size, test_image_size)

                                    processed_images = tf.expand_dims(processed_image, 0)

                                    logits, _ = network_fn(processed_images)
                                    predictions = tf.argmax(logits, 1)

                                    np_image, network_input, pose = sess.run([image_ex, processed_image, predictions])

                                    #pose = get_fp()
                                    print(pose)

                                    save_p = pic_save_path + other_name + '/' + str(pose) + '/' + pic_name + '_' + other_no + '.jpg' 

                                    cv2.imwrite(save_p,image)


    print(dict_learder_num.items())















'''


        with tf.Session() as sess:
            image = open(FLAGS.test_path, 'rb').read()
            image = tf.image.decode_jpeg(image, channels=3)

            processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)

            processed_images = tf.expand_dims(processed_image, 0)
            logits, _ = network_fn(processed_images)
            predictions = tf.argmax(logits, 1)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            np_image, network_input, predictions = sess.run([image, processed_image, predictions])
            print('{} {}'.format(FLAGS.test_path, predictions[0]))
'''
if __name__ == '__main__':
    tf.app.run()
