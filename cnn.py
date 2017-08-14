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
KERNEL_SIZE1=[16,3,3]
XSTRIDE=1
YSTRIDE=1
POOLSIZE=5
STEPS=100000
XSIZE=270
YSIZE=440
LEARNING_RATE=0.1
MODEL_ID="tsa23"
WEIGHTS=[1, 1]
DATA_PATH=os.environ["DATA_PATH"]

tf.logging.set_verbosity(tf.logging.INFO)

def build_model(data, labels, mode):
    data = tf.reshape(data, [BATCH_SIZE, 16, YSIZE, XSIZE, 1])
    conv1 = tf.layers.conv3d(inputs=data, filters=FILTER_COUNT, kernel_size=KERNEL_SIZE1, padding="same", strides=(1, XSTRIDE, YSTRIDE), activation=tf.nn.relu, name="conv1")
    print ("CONV " + str(conv1))
    tf.Print(conv1, [conv1], message="CONV1 ")
    conv2 = tf.layers.conv3d(inputs=conv1, filters=8, kernel_size=(16,3,3), padding="same", strides=(1, 2, 2), activation=tf.nn.relu, name="conv2")
    print ("CONV2 " + str(conv2))
    conv3 = tf.layers.conv3d(inputs=conv2, filters=8, kernel_size=(16,3,3), padding="same", strides=(1, 2, 2), activation=tf.nn.relu, name="conv3")
    print ("CONV3 " + str(conv3))
    pool1 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[1, POOLSIZE, POOLSIZE], strides=(1,2,2), name="pool1")
    print("POOL1" + str(pool1))
    conv4 = tf.layers.conv3d(inputs=pool1, filters=8, kernel_size=(16,5,5), padding="same", strides=(1, 2, 2), activation=tf.nn.relu, name="conv4")
    conv4=tf.identity(conv4, name="conv4")
    print ("CONV4 " + str(conv4))
    pool2 = tf.layers.max_pooling3d(inputs=conv4, pool_size=(1,4,4), strides=2, name="pool2")
    print("POOL2 " + str(pool2))
    flat_pool = tf.reshape(pool2, [BATCH_SIZE, 5376])
    sum_flat_pool = tf.reduce_sum(flat_pool) 
    sum_flat_pool=tf.identity(sum_flat_pool, name="sum_flat_pool")
    sum_data = tf.reduce_sum(data, axis=[1,2,3])
    sum_data=tf.identity(sum_data, name="sum_data")
    print ("FLAT POOL " + str(flat_pool))
    logits = tf.layers.dense(inputs=flat_pool, units=2)
    logits =tf.identity(logits, name="logits")
    logits = tf.reshape(logits, [BATCH_SIZE,2])
    print("LOGITS " + str(logits))
    #labels = tf.one_hot(labels, depth=2)
    print ("LABELS " + str(labels))
    flat_labels = tf.reshape(labels, [BATCH_SIZE, 2])
    #info = tf.concat([sum_data, tf.to_float(flat_labels)[:,0]], 1)
    #info = tf.stack([sum_data, tf.to_float(flat_labels)[:,0]], axis=1)
    #info = tf.identity(info, name="info")
    test_labels=tf.identity(flat_labels, name="labels")
    print("LABELS " + str(test_labels))
    class_weights=tf.reduce_sum(tf.multiply(flat_labels, tf.constant(WEIGHTS, dtype=tf.int64)), axis=1)
    print("CLASS WEIGHTS " + str(class_weights))
    loss = tf.losses.softmax_cross_entropy(onehot_labels=test_labels, logits=logits, weights=class_weights)
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


#tensors_to_log =  {"info":"info","probabilities": "softmax_tensor", "actual":"labels", "logits":"logits", "sum_flat_pool":"sum_flat_pool"}
tensors_to_log =  {"probabilities": "softmax_tensor", "actual":"labels", "logits":"logits", "sum_flat_pool":"sum_flat_pool"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5)

ids = image_df["id"].unique()[:29]
ids.sort()
print ("IDS " + str(ids))
def test_input():
    print("HERE")

#tsa_classifier.fit(x=tsa_utils.InputImagesIterator(ids, DATA_PATH, 10), y=tsa_utils.InputLabelsIterator(image_df, "Zone11",ids), steps=STEPS, batch_size=BATCH_SIZE, monitors=[logging_hook])

tsa_classifier.fit(x=tsa_utils.InputImagesIterator(ids, DATA_PATH, 10, "bottom", "right"), y=tsa_utils.InputLabelsIterator(image_df, "Zone11",ids), steps=STEPS, batch_size=BATCH_SIZE, monitors=[logging_hook])

filters = tsa_classifier.get_variable_value("conv2d/kernel")

print(filters.shape)

x = filters[:,:,0,0]
#plt.imshow(x)
#plt.show()


# In[21]:

dir(tf.python_io)


