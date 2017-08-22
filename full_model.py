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
import deep_cnn

DATA_PATH=os.environ["DATA_PATH"]
MODEL_ID=os.environ["MODEL_ID"]

image_df = pd.read_csv(DATA_PATH + '/stage1_labels.csv')
image_df['zone'] = image_df['Id'].str.split("_", expand=True)[1].str.strip()
image_df['id'] = image_df['Id'].str.split("_", expand=True)[0].str.strip()

ids = image_df["id"].unique()
ids.sort()

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
	16:("bottom","left")
}

for i in range(1,17):
	model = deep_cnn.ZoneModel(MODEL_ID, ids, "Zone" + str(i), slice_locations[i][1], slice_locations[i][0], DATA_PATH, image_df)
	model.train_model()