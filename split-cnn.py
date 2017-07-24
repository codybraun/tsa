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
BATCH_SIZE = 30
FILTER_COUNT= 16
KERNEL_SIZE1=[5,5]
XSTRIDE=3
YSTRIDE=3
POOLSIZE=5
STEPS=10000
XSIZE=512
YSIZE=660
LEARNING_RATE=.1
MODEL_ID="tsa3"
WEIGHTS=[1, 10]

tf.logging.set_verbosity(tf.logging.INFO)

def build_model(data, labels, mode):
    data = tf.reshape(data, [-1, 512, 660, 16])
    print(data.shape)
    conv1 = tf.layers.conv2d(inputs=data, filters=FILTER_COUNT, kernel_size=KERNEL_SIZE1, padding="same", strides=(XSTRIDE, YSTRIDE), activation=tf.nn.relu)
    print((XSIZE/XSTRIDE), (YSIZE/YSTRIDE), FILTER_COUNT)
    print ("CONV " + str(conv1))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[POOLSIZE, POOLSIZE], strides=POOLSIZE)
    print("POOL " + str(pool1))
    conv2 = tf.layers.conv2d(inputs=pool1, filters=FILTER_COUNT, kernel_size=2, padding="same", strides=(XSTRIDE, YSTRIDE), activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv2, filters=FILTER_COUNT, kernel_size=2, padding="same", strides=(XSTRIDE, YSTRIDE), activation=tf.nn.relu)
    print ("CONV2 " + str(conv2))
    print ("CONV3 " + str(conv3))
    pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2)
    print("POOL2 " + str(pool2))
    flat_pool = tf.reshape(pool2, [BATCH_SIZE, 20992])
    print ("FLAT POOL " + str(flat_pool))
    logits = tf.layers.dense(inputs=flat_pool, units=2)
    logits =tf.identity(logits, name="logits")
    logits = tf.reshape(logits, [BATCH_SIZE,2])
    print("LOGITS " + str(logits))
    print ("LABELS " + str(labels))
    flat_labels = tf.reshape(labels, [BATCH_SIZE, 2])
    test_labels=tf.identity(flat_labels, name="labels")
    print("LABELS " + str(test_labels))
    class_weights=tf.reduce_sum(tf.multiply(flat_labels, tf.constant(WEIGHTS, dtype=tf.int64)), axis=1)
    print("CLASS WEIGHTS " + str(class_weights))
    loss = tf.losses.softmax_cross_entropy(onehot_labels=test_labels, logits=logits, weights=class_weights)
    predictions = {
        "classes": tf.argmax(
          input=logits, axis=1, name="classes"),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")}
    print(predictions)
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=LEARNING_RATE,
        optimizer="SGD")
    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

image_df = pd.read_csv('/efs/stage1_labels.csv')
image_df['zone'] = image_df['Id'].str.split("_", expand=True)[1].str.strip()
image_df['id'] = image_df['Id'].str.split("_", expand=True)[0].str.strip()

tf.reset_default_graph()
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

tsa_classifier = tf.contrib.learn.Estimator(model_fn=build_model, model_dir="/tmp/" + MODEL_ID)


tensors_to_log = {"probabilities": "softmax_tensor", "classes":"classes", "actual":"labels", "logits":"logits"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5)

ids = image_df["id"].unique()
ids.sort()
print ("IDS " + str(ids))
tsa_classifier.fit(x=tsa_utils.InputImagesIterator(ids), y=tsa_utils.InputLabelsIterator(image_df, "Zone1",ids), steps=STEPS, batch_size=BATCH_SIZE, monitors=[logging_hook])

filters = tsa_classifier.get_variable_value("conv2d/kernel")

print(filters.shape)

x = filters[:,:,0,0]
#plt.imshow(x)
#plt.show()


# In[21]:

dir(tf.python_io)


