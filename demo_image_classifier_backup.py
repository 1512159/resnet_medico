
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import cv2
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import matplotlib
import numpy as np
matplotlib.use('Agg')
from matplotlib import pyplot as plt
slim = tf.contrib.slim
import glob
import os
CLASSES = [
    'blurry-nothing',
    'colon-clear',
    'dyed-lifted-polyps',
    'dyed-resection-margins',
    'esophagitis',
    'instruments',
    'normal-cecum',
    'normal-pylorus',
    'normal-z-line',
    'polyps',
    'retroflex-rectum',
    'retroflex-stomach',
    'stool-inclusions',
    'stool-plenty',
    'ulcerative-colitis']

tf.app.flags.DEFINE_integer(
    'batch_size', 150, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

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

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

def process_inp_image(img_name, demo_image_size,image_preprocessing_fn):
    image_data = tf.gfile.FastGFile(img_name, 'rb').read()
    image = tf.image.decode_jpeg(image_data, channels=3)
    processed_image = image_preprocessing_fn(image, demo_image_size, demo_image_size)
    processed_image  = tf.expand_dims(processed_image, 0)
    return processed_image
def main():
    if not FLAGS.dataset_dir:
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

       
        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)
        
        #####################################
        # Load image #
        #####################################
        
        # image_data = tf.gfile.FastGFile('test.jpg', 'rb').read()
        # image = tf.image.decode_jpeg(image_data, channels=3)
        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        print(preprocessing_name)
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name,is_training=False)
        demo_image_size = FLAGS.eval_image_size or network_fn.default_image_size
        
        # processed_image = image_preprocessing_fn(image, demo_image_size, demo_image_size)
        # processed_images  = tf.expand_dims(processed_image, 0)
        # items = glob.glob('/home/hoangtrunghieu/Medico2018/imdb/Medico_2018_test_set_cls/esophagitis/*.jpg')[:4 00]
        # images = []
        # images_name = []
        # items.sort()
        # for i, item in enumerate(items):
        #     images_name.append(os.path.splitext(os.path.basename(item))[0])
        #     images.append(process_inp_image(item, demo_image_size, image_preprocessing_fn))
            # if (i==10):
            #     sess = tf.Session()
                # processed_images  = tf.concat(images, 0)
                # [img_inp] = sess.run([processed_images])
                # print(img_inp.shape)
                # break
        # processed_images  = tf.concat(images, 0)
        test = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 224, 224, 3])
        # processed_images = tf.train.batch(test, batch_size=FLAGS.batch_size, enqueue_many=True, capacity=FLAGS.batch_size * 4, num_threads=3)
        
        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(test)
        # logits, _ = network_fn(processed_images)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)
        
        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
       
        model_variables = 'resnet_v2_50'
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))
        
        # qr=tf.train.QueueRunner(queue,[enqueue_op]*2)
        # tf.train.start_queue_runners(sess)

        RES_FILE = 'result.csv'
        if os.path.exists(RES_FILE):
            os.remove(RES_FILE)
        with tf.Session() as sess:
            items = glob.glob('/home/hoangtrunghieu/Medico2018/imdb/Medico_2018_test_set_cls/*/*.jpg')
            images = []
            images_name = []
            items.sort()
            # init_fn(sess)
            # print()
            for i, item in enumerate(items):
                images_name.append(os.path.splitext(os.path.basename(item))[0])
                images.append(process_inp_image(item, demo_image_size, image_preprocessing_fn))
                if (len(images) > 0) and (len(images) % FLAGS.batch_size == 0) or len(images) == len(items): 
                    if len(images) == len(items):
                        pdd = len(items) % FLAGS.batch_size
                        for _ in xrange(pdd):
                            images.append(np.zeros(dtype=np.float32,shape = [1, 224 ,224 , 3]))
                    processed_images  = tf.concat(images, 0)
                    inp_imgs = sess.run(processed_images)
                    print('Inp img done -> ', i + 1, len(items))
                    sess.run(tf.global_variables_initializer())
                    init_fn(sess)
                    tf.train.start_queue_runners(sess)
                    [prob, pred] = sess.run([probabilities, predictions],feed_dict = {test:inp_imgs})
                    fo = open(RES_FILE, "a")
                    for i, img_id in enumerate(images_name):
                        c = '{:s},{:s}'.format(img_id, str(CLASSES[pred[i]]))
                        _prob = prob[i]
                        # _cls_ind = np.argsort(-prob[i])
                        for k in range(16):
                            c+= ',{:3f}'.format(_prob[k])
                        fo.write(c+ '\n')
                    fo.close() 
                    images = []
                    images_name = []

if __name__ == '__main__':
    main()