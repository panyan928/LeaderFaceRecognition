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

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/disk3/py/tensorflow/models-master/models-master/research/slim/train_vgg_logs/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'test_path', 'fpface/frontal/f-0000000.jpg', 'Test image path.')

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

        tensor_input = tf.placeholder(tf.float32, [None, 112, 112, 3])

        tf.Graph().as_default()
        with tf.Session() as sess:            
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            images = list()

            #logits, _ = network_fn(processed_images)
            #predictions = tf.argmax(logits, 1)
            
            pic_path = '/home/disk3/py/face_data/pic_store/pre/dxp/'

            pic_path = '/home/disk3/py/face_data/pic_store/pre/dxp/DXP_03220_0.jpg'

            
            image = open(pic_path, 'rb').read()
            image = tf.image.decode_jpeg(image, channels=3)

            processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)

            processed_image = sess.run(processed_image)

            processed_images = tf.expand_dims(processed_image, 0)
            logits, _ = network_fn(processed_images)
            predictions = tf.argmax(logits, 1)
            images.append(processed_images)
            images = np.array(images)
            predictions = sess.run(logits, feed_dict = {tensor_input : images}).indices
            print(predictions)


            '''
            processed_images = tf.expand_dims(processed_image, 0)
            logits, _ = network_fn(processed_images)
            predictions = tf.argmax(logits, 1)

            np_image, network_input, predictions = sess.run([image, processed_image, predictions])
            print('{} {}'.format(pic_path, predictions[0]))

            pic_path = '/home/disk3/py/face_data/pic_store/pre/dxp/'

            pic_path = '/home/disk3/py/face_data/pic_store/pre/dxp/MZD_03350_0.jpg'

            
            image = open(pic_path, 'rb').read()
            image = tf.image.decode_jpeg(image, channels=3)

            processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)

            processed_images = tf.expand_dims(processed_image, 0)
            logits, _ = network_fn(processed_images)
            predictions = tf.argmax(logits, 1)
            #saver = tf.train.Saver()
            #saver.restore(sess, checkpoint_path)
            np_image, network_input, predictions = sess.run([image, processed_image, predictions])
            print('{} {}'.format(pic_path, predictions[0]))

            #pic_lists = glob.glob(pic_path+'/*.jpg')
            '''

            '''

            for pic_list in pic_lists:
                print(pic_list)
                image = open(pic_list, 'rb').read()
                image = tf.image.decode_jpeg(image, channels=3)

                processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)

                processed_images = tf.expand_dims(processed_image, 0)
                logits, _ = network_fn(processed_images)
                predictions = tf.argmax(logits, 1)
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_path)
                np_image, network_input, predictions = sess.run([image, processed_image, predictions])
                print('{} {}'.format(pic_list, predictions[0]))
            '''

if __name__ == '__main__':
    tf.app.run()
