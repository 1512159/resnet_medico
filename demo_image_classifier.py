# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from my_reporter import EvalReporter
from pycm import *

slim = tf.contrib.slim

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
    'out-of-patient',
    'polyps',
    'retroflex-rectum',
    'retroflex-stomach',
    'stool-inclusions',
    'stool-plenty',
    'ulcerative-colitis']

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


def _create_local(name, shape, collections=None, validate_shape=True,
                  dtype=tf.float32):
    """Creates a new local variable.
    Args:
      name: The name of the new or existing variable.
      shape: Shape of the new or existing variable.
      collections: A list of collection names to which the Variable will be added.
      validate_shape: Whether to validate the shape of the variable.
      dtype: Data type of the variables.
    Returns:
      The created variable.
    """
    # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
    collections = list(collections or [])
    collections += [tf.GraphKeys.LOCAL_VARIABLES]
    return tf.Variable(
        initial_value=tf.zeros(shape, dtype=dtype),
        name=name,
        trainable=False,
        collections=collections,
        validate_shape=validate_shape)


# Function to aggregate confusion
def _get_streaming_metrics(prediction, label, num_classes):
    with tf.name_scope("eval"):
        batch_confusion = tf.confusion_matrix(label, prediction,
                                              num_classes=num_classes,
                                              name='batch_confusion')
        
        confusion = _create_local('confusion_matrix',
                                  shape=[num_classes, num_classes],
                                  dtype=tf.int32)
        
        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = confusion.assign(tf.add(batch_confusion,confusion))
        # Cast counts to float so tf.summary.image renormalizes to [0,255]
        confusion_image = tf.reshape(tf.cast(confusion, tf.float32),
                                     [1, num_classes, num_classes, 1])

    return confusion, confusion_update

# def _get_pred_result(prediction):
#     with tf.name_scope("eval"):
#         predict = _create_local('my_predict',
#                                   shape=[100],
#                                   dtype=tf.int64)
#         # Create the update op for doing a "+=" accumulation on the batch
#         predict_update = predict.assign(prediction)
#         # predict_update = predict.assign(tf.concat(0,[v1, v2]))

#     return predict, predict_update


def main(_):
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
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label, img_id] = provider.get(['image', 'label', 'img_id'])
        label -= FLAGS.labels_offset
        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels, img_ids = tf.train.batch(
            [image, label, img_id],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=6 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(images)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()
        
        predictions = tf.argmax(logits, 1)
        probabilities = tf.nn.softmax(logits)
        scores = tf.maximum(probabilities,1)
        labels = tf.squeeze(labels)

        # result = tf.cond(tf.greater(probabilities[5], tf.constant(0.001)), lambda: tf.constant(1), lambda:tf.constant(2))

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 8),
            'Confusion_matrix': _get_streaming_metrics(labels, predictions,
                                                       dataset.num_classes - FLAGS.labels_offset)
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            if name in ['Confusion_matrix', 'Predictions'] :
                continue
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(
                dataset.num_samples / float(FLAGS.batch_size))
       
        print('Total samples: ', dataset.num_samples)
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        reporter = EvalReporter(img_ids, images, predictions, labels, probabilities) 
        reporter_op = reporter.get_op()

        model_variables = FLAGS.model_name
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))
        n = int(num_batches)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            sess.run(tf.global_variables_initializer())
            init_fn(sess)
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            all_pred = []
            all_label = []
            all_score = []
            all_id = []
            for _ in range(n):
                print('Predicting ' + str(_+1) + '/'+str(n))
                [ids, label, pred, score, _] = sess.run([img_ids, labels, predictions, probabilities, reporter_op ])
                #np array to list
                pred  = pred.tolist()
                label = label.tolist()
                id    = ids.tolist()
                score = score.tolist()

                #if instruments have score in top 3 and greater than 0.1 -> pred instruments
                for i in range(len(pred)):
                    if pred[i] in [2,3,10,6]:
                        dict_scores = {}
                        for j, scr in enumerate(score[i]):
                            dict_scores[j] = scr
                        sorted_d = sorted(dict_scores.items(), key=lambda x: x[1], reverse = True)
                        if ((sorted_d[1][0] == 5) and (sorted_d[1][1]>0.1)) or ((sorted_d[2][0] == 5) and (sorted_d[2][1]>0.1)) or ((sorted_d[3][0] == 5) and (sorted_d[3][1]>0.1) ):
                            pred[i] = 5
                
                #add to one big list
                all_pred += pred
                all_label+= label
                all_id   += id
                all_score+= score
            coord.request_stop()
            coord.join(threads)
        
        # Set all image with instrument score > 0.001
        # Save eval report    
        cm = ConfusionMatrix(actual_vector=all_label, predict_vector=all_pred)
        print(str(cm).split('\n')[43])
        with open(FLAGS.model_name + '_eval_result.txt',"w") as fo:
            fo.write(str(cm))
            fo.close()
        
        CHECK = set()
        #Save prediction
        with open(FLAGS.model_name + '_pred.csv',"w") as fo:
            fo.write('img_id,label,resnet_pred,resnet_score\n')
            for i in xrange(len(all_pred)):
                # if all_id[i] not in CHECK:
                    fo.write('{:s},{:s},{:s},{:s},{:s}\n'.format(all_id[i], 
                                                            CLASSES[all_label[i]],
                                                            CLASSES[all_pred[i]],
                                                            str(max(all_score[i])),
                                                            str(all_score[i]).replace('[','').replace(']','')
                                                            ))
                    # CHECK.add(all_id[i])
            fo.close()
        # print(confusion_matrix)
        # print(predicts)
        reporter.write_html_file(FLAGS.model_name + "_visualization.html")



if __name__ == '__main__':
    tf.app.run()
