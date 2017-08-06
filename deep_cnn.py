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
FILTER_COUNT= 8
KERNEL_SIZE1=[3,3]
XSTRIDE=1
YSTRIDE=1
POOLSIZE=2
STEPS=100000
XSIZE=270
YSIZE=440
LEARNING_RATE=0.01
MODEL_ID="tsa21"
WEIGHTS=[1, 1]
DATA_PATH=os.environ["DATA_PATH"]

tf.logging.set_verbosity(tf.logging.INFO)

def build_model(data, labels, mode):
    data = tf.reshape(data, [BATCH_SIZE, XSIZE, YSIZE, 16, 1])
    conv1 = tf.layers.conv3d(inputs=data, filters=8, kernel_size=[3,3,16], padding="same", strides=(1,1,1), activation=tf.nn.relu, name="conv1")
    pool0 = tf.layers.max_pooling3d(inputs=conv1, pool_size=(POOLSIZE, POOLSIZE, 1), strides=(2,2,1),  name="pool1")
    print ("CONV " + str(conv1))
    tf.Print(conv1, [conv1], message="CONV1 ")
    conv2 = tf.layers.conv3d(inputs=pool0, filters=8, kernel_size=[3,3,16], padding="same", strides= (1,1,1), activation=tf.nn.relu, name="conv2")
    print ("CONV2 " + str(conv2))
    pool1 = tf.layers.max_pooling3d(inputs=conv2, pool_size=(POOLSIZE, POOLSIZE, 1), strides=(1,1,1),  name="pool1")
    print("POOL1" + str(pool1))
    conv3 = tf.layers.conv3d(inputs=pool1, filters=FILTER_COUNT, kernel_size=[3,3,16], padding="same", strides=1, activation=tf.nn.relu, name="conv3")
    conv4 = tf.layers.conv3d(inputs=conv3, filters=FILTER_COUNT, kernel_size=[3,3,16], padding="same", strides=1, activation=tf.nn.relu, name="conv4")
    print ("CONV4 " + str(conv4))
    pool2 = tf.layers.max_pooling3d(inputs=conv4, pool_size=[3,3,1], strides=(2,2,1), name="pool2")
    conv5 = tf.layers.conv3d(inputs=pool2, filters=FILTER_COUNT, kernel_size=[3,3,16], padding="same", strides=1, activation=tf.nn.relu, name="conv5")
    print ("CONV5 " + str(conv5))
    pool3 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[3,3,1], strides=(1,1,1), name="pool2")
    conv6 = tf.layers.conv3d(inputs=pool3, filters=FILTER_COUNT, kernel_size=[3,3,16], padding="same", strides=1, activation=tf.nn.relu, name="conv6")
    conv7 = tf.layers.conv3d(inputs=conv6, filters=FILTER_COUNT, kernel_size=[3,3,16], padding="same", strides=1, activation=tf.nn.relu, name="conv7")
    print ("CONV7 " + str(conv7))
    pool2 = tf.layers.max_pooling3d(inputs=conv7, pool_size=[3,3,1], strides=(1,1,1), name="pool2")

    print("POOL2 " + str(pool2))
    flat_pool = tf.reshape(pool2, [BATCH_SIZE, 833280])
    sum_flat_pool = tf.reduce_sum(flat_pool) 
    sum_flat_pool=tf.identity(sum_flat_pool, name="sum_flat_pool")
    print ("FLAT POOL " + str(flat_pool))
    #dense_layer = tf.layers.dense(inputs=flat_pool, activation=tf.nn.relu, units=200)
    #print ("DENSE LAYER " + str(dense_layer))
    logits = tf.layers.dense(inputs=flat_pool, units=2)
    logits =tf.identity(logits, name="logits")
    logits = tf.reshape(logits, [BATCH_SIZE,2])
    #logits = tf.add(logits, ([([0.000001] * 2)] * BATCH_SIZE))
    print("LOGITS " + str(logits))
    #labels = tf.one_hot(labels, depth=2)
    print ("LABELS " + str(labels))
    flat_labels = tf.reshape(labels, [BATCH_SIZE, 2])
    test_labels=tf.identity(flat_labels, name="labels")
    print("LABELS " + str(test_labels))
    class_weights=tf.reduce_sum(tf.multiply(flat_labels, tf.constant(WEIGHTS, dtype=tf.int64)), axis=1)
    print("CLASS WEIGHTS " + str(class_weights))
    loss = tf.losses.softmax_cross_entropy(onehot_labels=test_labels, logits=logits, weights=class_weights)
    #loss = tf.losses.log_loss(test_labels, logits)
    #loss = tf.losses.mean_squared_error(test_labels, logits)
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

image_df = pd.read_csv(DATA_PATH + '/stage1_labels.csv')
image_df['zone'] = image_df['Id'].str.split("_", expand=True)[1].str.strip()
image_df['id'] = image_df['Id'].str.split("_", expand=True)[0].str.strip()

tf.reset_default_graph()
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

tsa_classifier = tf.contrib.learn.Estimator(model_fn=build_model, model_dir="/tmp/" + MODEL_ID)


tensors_to_log =  {"probabilities": "softmax_tensor", "classes":"classes", "actual":"labels", "logits":"logits", "sum_flat_pool":"sum_flat_pool"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5)

ids = image_df["id"].unique()
ids = ids[:30]
ids.sort()
def test_input():
    print("HERE")

tsa_classifier.fit(x=tsa_utils.InputImagesIterator(ids, DATA_PATH, 10, "bottom", "right"), y=tsa_utils.InputLabelsIterator(image_df, "Zone11",ids), steps=STEPS, batch_size=BATCH_SIZE, monitors=[logging_hook])

filters = tsa_classifier.get_variable_value("conv2d/kernel")

print(filters.shape)

x = filters[:,:,0,0]
#plt.imshow(x)
#plt.show()


# In[21]:

dir(tf.python_io)


