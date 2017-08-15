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
POOLSIZE1=(1,3,3)
POOL_STRIDES=(2,2,2)
STEPS=1000000
XSIZE=270
YSIZE=340
LEARNING_RATE=0.001
IMAGE_DEPTH=16
CHANNELS=1
MODEL_ID=os.environ["MODEL_ID"]
WEIGHTS=[1, 1]
DATA_PATH=os.environ["DATA_PATH"]

tf.logging.set_verbosity(tf.logging.INFO)

def build_model(data, labels, mode):
    data = tf.reshape(data, [BATCH_SIZE, IMAGE_DEPTH, YSIZE, XSIZE, CHANNELS])
    conv1 = tf.layers.conv3d(inputs=data, filters=FILTER_COUNT, kernel_size=KERNEL_SIZE1, padding="same", strides=(DEPTHSTRIDE,XSTRIDE,YSTRIDE), name="conv1")
    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=POOLSIZE1, strides=POOL_STRIDES, name="pool1")
    conv2 = tf.layers.conv3d(inputs=pool1, filters=FILTER_COUNT, kernel_size=(2,3,3), padding="same", strides=(1, 1, 1), name="conv2", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=POOLSIZE1, strides=POOL_STRIDES, name="pool2")
    conv3 = tf.layers.conv3d(inputs=pool2, filters=FILTER_COUNT, kernel_size=KERNEL_SIZE1, padding="same", strides=(DEPTHSTRIDE,XSTRIDE,YSTRIDE), name="conv3")
    pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=POOLSIZE1, strides=POOL_STRIDES, name="pool1")
    conv4 = tf.layers.conv3d(inputs=pool3, filters=FILTER_COUNT, kernel_size=(2,3,3), padding="same", strides=(1, 1, 1), name="conv4", activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling3d(inputs=conv4, pool_size=POOLSIZE2, strides=POOL_STRIDES, name="pool2")
    print("DATA " + str(data))
    print ("CONV2 " + str(conv2))
    print ("pool1 " + str(pool1))
    print ("CONV " + str(conv1))
    print("POOL2 " + str(pool2))
    print("CONV3 " + str(conv3))
    print("POOL3 " + str(pool3))
    print("CONV4 " + str(conv4))
    print("POOL4 " + str(pool4))

    flat_pool = tf.reshape(pool2, [BATCH_SIZE, 133056])
    flat_pool=tf.identity(flat_pool, name="flat_pool")
    sum_conv1 = tf.reduce_sum(conv1, axis=[1,2,3,4]) 
    sum_conv1=tf.identity(sum_conv1, name="sum_conv1")
    sum_pool1 = tf.reduce_sum(pool1, axis=[1,2,3,4]) 
    sum_pool1=tf.identity(sum_pool1, name="sum_pool1")
    sum_conv2 = tf.reduce_sum(conv2, axis=[1,2,3,4]) 
    sum_conv2=tf.identity(sum_conv2, name="sum_conv2")
    sum_pool2 = tf.reduce_sum(pool2, axis=[1,2,3,4]) 
    sum_pool2 = tf.identity(sum_pool2, name="sum_pool2")
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

image_df = pd.read_csv(DATA_PATH + '/stage1_labels.csv')
image_df['zone'] = image_df['Id'].str.split("_", expand=True)[1].str.strip()
image_df['id'] = image_df['Id'].str.split("_", expand=True)[0].str.strip()

tsa_classifier = tf.contrib.learn.Estimator(model_fn=build_model, model_dir="/tmp/" + MODEL_ID)

tensors_to_log =  {"sum_flat_pool":"sum_flat_pool","sum_data":"sum_data","sum_conv1":"sum_conv1","sum_conv2":"sum_conv2","sum_pool1":"sum_pool1","sum_pool2":"sum_pool2","flat_pool":"flat_pool","probabilities": "softmax_tensor", "actual":"labels", "logits":"logits"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5)

ids = image_df["id"].unique()
ids.sort()

tsa_classifier.fit(x=tsa_utils.InputImagesIterator(ids, DATA_PATH, 10000, "bottom", "right"), y=tsa_utils.InputLabelsIterator(image_df, "Zone11",ids), steps=STEPS, batch_size=BATCH_SIZE, monitors=[logging_hook])
