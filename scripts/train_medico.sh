#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifarnet_on_cifar10.sh
set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=medico-model-kvasir-v2
# Where the dataset is saved to.
DATASET_DIR=dataset-kvasir-v2
# Where the test dataset is saved to
TESTSET_DIR=medico-test-dataset 

# Test dataset
# python2 download_and_convert_data.py \
#   --dataset_name=medico_test \
#   --dataset_dir=${TESTSET_DIR}


# Train_val dataset
# python2 download_and_convert_data.py \
#   --dataset_name=medico \
#   --dataset_dir=${DATASET_DIR}

# Run training.
# python2 train_image_classifier.py \
#   --train_dir=${TRAIN_DIR} \
#   --dataset_name=medico \
#   --dataset_split_name=train \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=resnet_v2_50 \
#   --preprocessing_name=medico_preprocessing \
#   --max_number_of_steps=100000 \
#   --batch_size=64 \
#   --save_interval_secs=120 \
#   --save_summaries_secs=120 \
#   --log_every_n_steps=100 \
#   --optimizer=sgd \
#   --learning_rate=0.00001 \
#   --learning_rate_decay_factor=0.1 \
#   --num_epochs_per_decay=200 \
#   --weight_decay=0.004

# Run evaluation.
# python2 eval_image_classifier.py \
#   --checkpoint_path=${TRAIN_DIR} \
#   --eval_dir=${TRAIN_DIR} \
#   --dataset_name=medico \
#   --dataset_split_name=validation \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=resnet_v2_50 \
#   --preprocessing_name=medico_preprocessing

python2 demo_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=medico \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v2_50 \
  --preprocessing_name=medico_preprocessing
