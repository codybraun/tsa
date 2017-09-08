import numpy as np
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import matplotlib.animation
import tensorflow as tf
from numpy import genfromtxt
import pandas as pd
import sys
sys.path.append(os.getcwd())
import tsa_utils

#Model parameters
BATCH_SIZE=30
FILTER_COUNT=8
KERNEL_SIZE1=(16,3,3)
DEPTHSTRIDE=1
XSTRIDE=1
YSTRIDE=1
POOLSIZE1=3
POOLSIZE2=(1,3,3)
POOL_STRIDES=(2,2,2)
STEPS=20000
XSIZE=270
YSIZE=340
LEARNING_RATE=0.001
IMAGE_DEPTH=16
CHANNELS=1
WEIGHTS=[1, 1]

tf.logging.set_verbosity(tf.logging.INFO)


class ZoneModel():

    def __init__(self, model_id, ids, zone, x_slice, y_slice, data_path, labels, checkpoint_path="."):
        self.model_id = model_id
        self.ids = ids
        self.zone = zone if zone else ""
        self.x_slice = x_slice
        self.y_slice = y_slice
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.labels = labels

    def build_model(self, data, labels, mode):
        print(data, labels, mode)
        if mode == tf.contrib.learn.ModeKeys.INFER:
            BATCH_SIZE=1
        else:
            BATCH_SIZE=30
        data = tf.reshape(data, [BATCH_SIZE, IMAGE_DEPTH, YSIZE, XSIZE, CHANNELS])
        conv1 = tf.layers.conv3d(inputs=data, filters=FILTER_COUNT, kernel_size=KERNEL_SIZE1, padding="same", strides=(DEPTHSTRIDE,XSTRIDE,YSTRIDE), name="conv1")
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=POOLSIZE1, strides=POOL_STRIDES, name="pool1")
        conv2 = tf.layers.conv3d(inputs=pool1, filters=FILTER_COUNT, kernel_size=(2,3,3), padding="same", strides=(1, 1, 1), name="conv2", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=POOLSIZE1, strides=POOL_STRIDES, name="pool2")
        conv3 = tf.layers.conv3d(inputs=pool2, filters=FILTER_COUNT, kernel_size=KERNEL_SIZE1, padding="same", strides=(DEPTHSTRIDE,XSTRIDE,YSTRIDE), name="conv3")
        pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=POOLSIZE1, strides=POOL_STRIDES, name="pool1")
        conv4 = tf.layers.conv3d(inputs=pool3, filters=FILTER_COUNT, kernel_size=(2,3,3), padding="same", strides=(1, 1, 1), name="conv4", activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling3d(inputs=conv4, pool_size=POOLSIZE2, strides=POOL_STRIDES, name="pool2")
        
        flat_pool = tf.reshape(pool2, [BATCH_SIZE, 133056])
        flat_pool=tf.identity(flat_pool, name="flat_pool")

        logits = tf.layers.dense(inputs=flat_pool, units=2)
        logits =tf.identity(logits, name="logits")
        logits = tf.reshape(logits, [BATCH_SIZE,2])

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            flat_labels = tf.reshape(labels, [BATCH_SIZE, 2])
            test_labels=tf.identity(flat_labels, name="labels")
            class_weights=tf.reduce_sum(tf.multiply(flat_labels, tf.constant(WEIGHTS, dtype=tf.int64)), axis=1)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=test_labels, logits=logits, weights=class_weights)
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=LEARNING_RATE,
                optimizer="SGD")
        predictions = {
            "classes": tf.argmax(
              input=logits, axis=1, name="classes"),
          "probabilities": tf.nn.softmax(
              logits, name="softmax_tensor")}
        if mode == tf.contrib.learn.ModeKeys.INFER:
            return tf.contrib.learn.ModelFnOps(mode=mode, predictions=predictions)
        return tf.contrib.learn.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

    def train_model(self, tensors_to_log):
        tsa_classifier = tf.contrib.learn.Estimator(model_fn=self.build_model, 
                                                    model_dir=self.checkpoint_path + "/" + self.model_id + self.zone)
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)
        tsa_classifier.fit(
            x=tsa_utils.InputImagesIterator(self.ids, self.data_path, 10000, self.y_slice, self.x_slice), 
            y=tsa_utils.InputLabelsIterator(self.ids, self.labels), 
            steps=STEPS, 
            batch_size=BATCH_SIZE, 
            monitors=[logging_hook])
        self.model = tsa_classifier

    def load_model(self):
        tsa_classifier = tf.contrib.learn.Estimator(model_fn=self.build_model, 
                                                    model_dir=self.checkpoint_path + "/" + self.model_id + self.zone)
        print ("LOADED MODEL AT STEP " + str(tsa_classifier.get_variable_value("global_step")))
        self.model = tsa_classifier

    def bootstrap_model(self):
        tsa_classifier = tf.contrib.learn.Estimator(model_fn=self.build_model, 
                                                    model_dir=self.checkpoint_path)
        self.model = tsa_classifier

    def predict(self):
        return self.model.predict(x=tsa_utils.InputImagesIterator(self.ids, 
                                                            self.data_path, 
                                                            10000, 
                                                            self.y_slice,
                                                            self.x_slice, False))