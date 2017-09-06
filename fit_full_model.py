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
import mini_cnn


DATA_PATH=os.environ["DATA_PATH"]
CHECKPOINT_PATH=os.environ["CHECKPOINT_PATH"]
MODEL_ID=os.environ["MODEL_ID"]

image_df = pd.read_csv(DATA_PATH + '/stage1_labels.csv')
image_df['zone'] = image_df['Id'].str.split("_", expand=True)[1].str.strip()
image_df['id'] = image_df['Id'].str.split("_", expand=True)[0].str.strip()

ids = image_df["id"].unique()
ids.sort()
training_ids = ids

labels = image_df[image_df['id'].isin(training_ids)]
labels = labels.sort_values("id")

slice_locations = {1:("top","right"),
	2:("middle","right"), 
	3:("top","left"),
	4:("middle","left"),
	5:("top","middle"),
	6:("top","right"),
	7:("top","left"),
	8:("middle","right"),
	9:("middle","middle"),
	10:("middle","left"),
	11:("bottom","right"),
	12:("bottom","left"),
	13:("bottom","right"),
	14:("bottom","left"),
	15:("bottom","right"),
	16:("bottom","left"),
	17:("top","middle")
}
tensors_to_log =  {"probabilities": "softmax_tensor",
                    "actual":"labels"}

with tf.Session() as sess:
	for zone in range(17,17):
		print("FITTING MODEL FOR ZONE " + str(zone))
		train_labels = labels[labels["zone"]=="Zone"+str(zone)]
		train_labels["class0"] = 0
		train_labels["class1"] = 0
		train_labels.loc[train_labels['Probability'] == 0, 'class0'] = 1
		train_labels.loc[train_labels['Probability'] == 1, 'class1'] = 1
		train_labels = np.reshape(np.array(train_labels[["class0","class1"]]), [-1,2])
		model = deep_cnn.ZoneModel(MODEL_ID, training_ids, "Zone" + str(zone), slice_locations[zone][1], slice_locations[zone][0], DATA_PATH, train_labels, CHECKPOINT_PATH)
		model.bootstrap_model()
		model.train_model(tensors_to_log)


	