
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
tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

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

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

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

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        # provider = slim.dataset_data_provider.DatasetDataProvider(
        #     dataset,
        #     shuffle=False,
        #     common_queue_capacity=2 * FLAGS.batch_size,
        #     common_queue_min=FLAGS.batch_size)
        # [image, label] = provider.get(['image', 'label'])
        # label -= FLAGS.labels_offset
        
        #####################################
        # Load image #
        #####################################
        # items = glob.glob('/home/hoangtrunghieu/Medico2018/imdb/Medico_2018_test_set_cls/esophagitis/*.jpg')[:3]
        file_name = tf.constant('test.jpg')
        image_data = tf.gfile.FastGFile(file_name, 'rb').read()
        image = tf.image.decode_jpeg(image_data, channels=3)
        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name,is_training=False)
        demo_image_size = FLAGS.eval_image_size or network_fn.default_image_size
        processed_image = image_preprocessing_fn(image, demo_image_size, demo_image_size)
        processed_images  = tf.expand_dims(processed_image, 0)
        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(processed_images)
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
        
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        # init_fn(sess)
        # print(sess.run(image_data, feed_dict = {file_name: 'test.jpg'}))
        # [a] = sess.run([probabilities], feed_dict = {file_name: 'test.jpg'})
        # print(a)
            # items = glob.glob('/home/hoangtrunghieu/Medico2018/imdb/Medico_2018_test_set_cls/esophagitis/*.jpg')
            # for item in items:
                # image_data = tf.gfile.FastGFile(item, 'rb').read()
                # image = tf.image.decode_jpeg(image_data, channels=3)
                # processed_image = image_preprocessing_fn(image, demo_image_size, demo_image_size)
                # processed_images  = tf.expand_dims(processed_image, 0)
                # logits, _ = network_fn(processed_images)
                # probabilities = tf.nn.softmax(logits)
                # predictions = tf.argmax(logits, 1)
                # [a] = sess.run([probabilities])
                # print(a)
            
            # network_input, probabilities = sess.run([processed_images, probabilities])
                                    
            
            # probabilities = probabilities[0, 0:]
            # sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
            #                                     key=lambda x:x[1])]
            
            # np_image, network_input = sess.run([image,processed_image])
            # plt.figure()
            # plt.imshow(np_image[0].astype(np.uint8))
            # plt.axis('off')
            # plt.savefig('inp.jpg')

            # plt.figure()
            # print(np_image.shape)
            # print(network_input.shape)
            # plt.imshow( network_input / (network_input.max() - network_input.min()) )
            # plt.axis('off')
            # plt.savefig('network-input.jpg')
            # for i in range(5):
            #     index = sorted_inds[i]
            #     print('Probability %0.2f => [%s]' % (probabilities[index], str(index)))
    
    # res = slim.get_model_variables()
if __name__ == '__main__':
    main()