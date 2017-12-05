import os
import numpy as np
import pickle
import skimage
import skimage.measure
import skimage.io
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale as sknorm
from sklearn.preprocessing import scale
import random
from pathlib import Path

def read_header(infile):
    """Read image header (first 512 bytes)
    """
    h = dict()
    fid = open(infile, 'rb')
    h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
    h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
    h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
    h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)
    return h

def flip_image(data):
    flipped = np.flip(data, axis=2)
    flipped = np.flip(flipped, axis=0)
    return flipped

def read_data(infile, threshold=.00005, normalize=True, use_cache=True, cache_path="/tmp/", pool_size=1):
    if "reverse" in infile:
        reverse= True
    else:
        reverse=False
    base_file, extension = os.path.splitext(os.path.basename(infile))
    if use_cache:
        cached_file = Path(cache_path + base_file + ".p")
        if cached_file.is_file():
            data = pickle.load(open(cache_path + base_file + ".p", "rb"))
            return data
    infile = infile.replace("reverse","")
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512) #skip header
    if extension == '.aps':
        angles = 16
    else:
        angles=64
    if extension == '.aps' or extension == '.a3daps':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
    fid.close()
    if normalize:
        data = (data- data.mean()) / data.std()
    data[data<threshold] = 0
    data = np.swapaxes(data, 2, 0)
    data = skimage.measure.block_reduce(data.reshape(-1, 660, 512, order="A"), (1,pool_size,pool_size), np.mean).astype(np.float32)
    #data = np.swapaxes(data, 0, 1)
    if reverse:
        data = flip_image(data)
    if use_cache:
        pickle.dump(data, open(cache_path + base_file + ".p", "wb+"))
    return data

def load_thresholded_image(image_path, x=11, y=5):
    data = read_data(image_path).reshape(16 * 660, 512, order="A")
    data *= 255/data.max()
    data = data.astype(np.uint8)
    return cv2.adaptiveThreshold(data,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,x,y)

def next_iter(self):
    if self.i < len(self.ids) -1:
        self.i = self.i + 1
        return self.calculate_data()
    else:
        if not self.repeating:
            raise StopIteration()
            self.i = -1
            return self.calculate_data()

class InputImagesIterator:
    def __init__(self, ids, data_path, repeating=True, pool_size=1, scaler_ids=None, file_format=".aps", seed=111, randomize=False, normalize=True):
        self.ids=ids
        self.data_path=data_path
        self.i = -1
        self.repeating = repeating
        self.pool_size = pool_size
        self.file_format = file_format
        self.seed=seed
        self.randomize=randomize
        self.local_random = random.Random()
        self.local_random.seed(seed)
        self.normalize=normalize

    def __iter__(self):
        return self

    def calculate_data(self):
        if self.randomize:
            j = self.local_random.randint(-1,len(self.ids)-1)
        else:
            j = self.i
        data = read_data(str(self.data_path + str(self.ids[j]) + str(self.file_format)), normalize=self.normalize, pool_size=self.pool_size)
        return data

    def __next__(self):
        return next_iter(self)

class SampledImagesIterator:
    def __init__(self, ids, data_path, repeating=True, pool_size=1, file_format=".aps", seed=111, randomize=False, sample=True):
        self.ids=ids
        self.data_path=data_path
        self.i = -1
        self.repeating = repeating
        self.pool_size = pool_size
        self.file_format = file_format
        self.seed=seed
        self.randomize=randomize
        self.local_random = random.Random()
        self.local_random.seed(seed)
        self.sample = sample

    def __iter__(self):
        return self
    
    def calculate_data(self):
        if self.randomize:
            j = self.local_random.randint(-1,len(self.ids)-1)
        else:
            j = self.i
        data = read_data(str(self.data_path + str(self.ids[j]) + str(self.file_format)), pool_size=self.pool_size)
        return self.sample_data(data)

    def __next__(self):
        return next_iter(self)
        
    def sample_data(self, data):
        if self.sample:
            rands = np.random.randint(3, size=16)
            aug = np.arange(start=0, stop=64, step=4)
            indices = rands + aug
            data =data[indices,:,:]
            return data
        else:
            return data[(self.i%3)::4,:,:]

class InputLabelsIterator:
    def __init__(self, ids, labels, randomize=False, seed=111):
        self.ids=ids
        self.labels=labels
        self.i=-1
        self.randomize = randomize
        self.seed=seed
        self.local_random = random.Random()
        self.local_random.seed(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < len(self.ids) -1:
            self.i = self.i + 1
            if self.randomize:
                j = self.local_random.randint(-1,len(self.ids)-1)
            else:
                j = self.i
            return(self.labels[j])
        else:
            self.i = -1
            if self.randomize:
                j = self.local_random.randint(-1,len(self.ids)-1)
            else:
                j = self.i
            #raise StopIteration()
            return(self.labels[j])
