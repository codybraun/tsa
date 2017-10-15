import numpy as np
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import matplotlib.animation
import tensorflow as tf
from numpy import genfromtxt
import pandas as pd
import sys
import math 

sys.path.append(os.getcwd())
import tsa_utils
import deep_cnn
import flattened_cnn2
import mini_cnn


DATA_PATH=os.environ["DATA_PATH"]
CHECKPOINT_PATH=os.environ["CHECKPOINT_PATH"]
MODEL_ID=os.environ["MODEL_ID"]

image_df = pd.read_csv(DATA_PATH + '/stage1_labels.csv')
image_df['zone'] = image_df['Id'].str.split("_", expand=True)[1].str.strip()
image_df['id'] = image_df['Id'].str.split("_", expand=True)[0].str.strip()
image_df = image_df.groupby(["id"]).sum().reset_index()
image_df.loc[image_df['Probability'] == 0, 'probability'] = 0
image_df.loc[image_df['Probability'] >= 1, 'probability'] = 1

ids = image_df["id"].unique()
ids.sort()
training_ids = ids

labels = image_df[image_df['id'].isin(training_ids)]
train_labels = labels.sort_values("id")

tensors_to_log =  {"probabilities": "softmax_tensor",
                    "actual":"labels"}

with tf.Session() as sess:

	train_labels["class0"] = 0
	train_labels["class1"] = 0
	train_labels.loc[train_labels['Probability'] == 0, 'class0'] = 1
	train_labels.loc[train_labels['Probability'] > 0, 'class1'] = 1
	train_labels = np.reshape(np.array(train_labels[["class0","class1"]]), [-1,2])
	# model = flattened_cnn.ZoneModel(MODEL_ID, training_ids, None, None, None, DATA_PATH, train_labels, CHECKPOINT_PATH)
	model = flattened_cnn2.ZoneModel(MODEL_ID, training_ids, None, None, None, DATA_PATH, train_labels, CHECKPOINT_PATH)
	#model.bootstrap_model()
	model.train_model(tensors_to_log, reuse=False)


	