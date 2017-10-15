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
import sklearn
from sklearn import metrics

sys.path.append(os.getcwd())
import tsa_utils
import flattened_cnn2
import mini_cnn

DATA_PATH=os.environ["DATA_PATH"]
MODEL_ID=os.environ["MODEL_ID"]
CHECKPOINT_PATH=os.environ["CHECKPOINT_PATH"]

image_df = pd.read_csv(DATA_PATH + 'stage1_sample_submission.csv')
image_df['zone'] = image_df['Id'].str.split("_", expand=True)[1].str.strip()
image_df['id'] = image_df['Id'].str.split("_", expand=True)[0].str.strip()

ids = image_df["id"].unique()
ids.sort()
labels = image_df[image_df['id'].isin(ids)]
all_labels = labels.sort_values("id")

csv_file = open(CHECKPOINT_PATH + "/submission.csv", "w+")
csv_file.write("Id,Probability\n")
for zone in range(1,18):
	print("EVALUATING MODEL FOR ZONE " + str(zone))
	labels = all_labels[all_labels["zone"]=="Zone"+str(zone)]
	labels["class0"] = 0
	labels["class1"] = 0
	labels.loc[labels['Probability'] == 0, 'class0'] = 1
	labels.loc[labels['Probability'] == 1, 'class1'] = 1
	labels = np.reshape(np.array(labels[["class0","class1"]]), [-1,2])

	#model = deep_cnn.ZoneModel(MODEL_ID, ids, "Zone" + str(i), slice_locations[i][1], slice_locations[i][0], DATA_PATH, image_df)
	model = flattened_cnn2.ZoneModel(MODEL_ID, ids, "Zone" + str(zone), "both", "both", DATA_PATH, labels, checkpoint_path=CHECKPOINT_PATH)
	model.load_model()
	print ("LOADED MODEL " + str(model))
	predicted = np.array([x["probabilities"][1] for x in list(model.predict())])
	for i, prediction in enumerate(predicted):
		csv_file.write(str(ids[i]) + "_Zone" + str(zone) + "," + str(prediction) + "\n")
csv_file.close()



