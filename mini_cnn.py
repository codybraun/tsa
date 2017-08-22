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
BATCH_SIZE=5
FILTER_COUNT=1
KERNEL_SIZE1=(3,3,3)
DEPTHSTRIDE=5
XSTRIDE=5
YSTRIDE=5
POOLSIZE1=3
POOLSIZE2=(1,3,3)
POOL_STRIDES=(3,3,3)
STEPS=10
XSIZE=270
YSIZE=340
LEARNING_RATE=0.001
IMAGE_DEPTH=16
CHANNELS=1
WEIGHTS=[1, 1]

tf.logging.set_verbosity(tf.logging.INFO)


class ZoneModel():

    def __init__(self, model_id, ids, zone, x_slice, y_slice, data_path, labels):
        self.model_id = model_id
        self.ids = ids
        self.zone = zone
        self.x_slice = x_slice
        self.y_slice = y_slice
        self.data_path = data_path
        self.labels = labels

    def build_model(self, data, labels, mode):
        print(data)
        data = tf.reshape(data, [BATCH_SIZE, IMAGE_DEPTH, YSIZE, XSIZE, CHANNELS])
        dropout = tf.layers.dropout(inputs=data, rate=0.5)
        conv1 = tf.layers.conv3d(inputs=dropout, 
                filters=FILTER_COUNT, 
                kernel_size=KERNEL_SIZE1, 
                padding="valid", 
                strides=(DEPTHSTRIDE,XSTRIDE,YSTRIDE), 
                name="conv1")
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=POOLSIZE1, strides=POOL_STRIDES, name="pool1")
        print(conv1)
        print(pool1)
        flat_pool = tf.reshape(pool1, [BATCH_SIZE, 396])
        flat_pool=tf.identity(flat_pool, name="flat_pool")
        sum_conv1 = tf.reduce_sum(conv1, axis=[1,2,3,4]) 
        sum_conv1=tf.identity(sum_conv1, name="sum_conv1")
        sum_pool1 = tf.reduce_sum(pool1, axis=[1,2,3,4]) 
        sum_pool1=tf.identity(sum_pool1, name="sum_pool1")
    
        sum_flat_pool = tf.reduce_sum(flat_pool, axis=[1]) 
        sum_flat_pool = tf.identity(sum_flat_pool, name="sum_flat_pool")

        sum_data = tf.reduce_sum(data, axis=[1,2,3])
        sum_data=tf.identity(sum_data, name="sum_data")
        logits = tf.layers.dense(inputs=flat_pool, units=2)
        logits =tf.identity(logits, name="logits")
        logits = tf.reshape(logits, [BATCH_SIZE,2])

        flat_labels = tf.reshape(labels, [BATCH_SIZE, 2])
        test_labels=tf.identity(flat_labels, name="labels")
        class_weights=tf.reduce_sum(tf.multiply(flat_labels, tf.constant(WEIGHTS, dtype=tf.int64)), axis=1)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=test_labels, logits=logits, weights=class_weights)
        predictions = {
            "classes": tf.argmax(
              input=logits, axis=1, name="classes"),
          "probabilities": tf.nn.softmax(
              logits, name="softmax_tensor")}
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=LEARNING_RATE,
            optimizer="SGD")

        return tf.contrib.learn.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

    def train_model(self):
        tsa_classifier = tf.contrib.learn.Estimator(model_fn=self.build_model, model_dir="/output/" + self.model_id + self.zone)
        tensors_to_log =  {"sum_flat_pool":"sum_flat_pool",
                            "sum_data":"sum_data",
                            "sum_conv1":"sum_conv1",
                            "sum_pool1":"sum_pool1",
                            "flat_pool":"flat_pool",
                            "probabilities": "softmax_tensor",
                            "actual":"labels", "logits":"logits"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5)

        tsa_classifier.fit(
                x=tsa_utils.InputImagesIterator(self.ids, self.data_path, 10000, self.y_slice, self.x_slice), 
                y=tsa_utils.InputLabelsIterator(self.ids, self.labels), 
                steps=STEPS, 
                batch_size=BATCH_SIZE, 
                monitors=[logging_hook])
