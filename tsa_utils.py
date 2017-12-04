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

def read_data(infile, vertical="both", horizontal="both", threshold=.00005, normalize=True, use_cache=True, cache_path="/tmp/", pool_size=1):
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
        data = (data- data.mean()) / x.std()
    data[data<threshold] = 0
    data = np.swapaxes(data, 2, 0)
    data = skimage.measure.block_reduce(data.reshape(-1, 660, 512, order="A"), (1,pool_size,pool_size), np.mean).astype(np.float32)
    #data = np.swapaxes(data, 0, 1)
    if reverse:
        data = flip_image(data)
    if use_cache:
        pickle.dump(data, open(cache_path + base_file + ".p", "wb+"))
    return data
   
def sample_data(data):
    rands = np.random.randint(3, size=16)
    aug = np.arange(start=0, stop=64, step=4)
    indices = rands + aug
    data =data[indices,:,:]
    return data
    
def spread_spectrum(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img= clahe.apply(img)
    return img

def convert_to_grayscale(img):
    base_range = np.amax(img) - np.amin(img)
    rescaled_range = 255 - 0
    img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)
    img_rescaled[img_rescaled < 12]=0
    return np.uint8(img_rescaled)

def normalize(image):
    MIN_BOUND = 0.0
    MAX_BOUND = 255.0
    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
     
    image_mean = image.mean()
    
    image = image - image_mean
    return image

def read_data_coords(infile, x, y, x_size, y_size):
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512) #skip header
    if(h['word_type']==7): #float32
        data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
    elif(h['word_type']==4): #uint16
        data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
    data = data * h['data_scale_factor'] #scaling factor
    data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
    data = data[x:x + x_size, y:y+y_size, :]
    fid.close()
    #return(data)
    return np.swapaxes(data, 2, 0)

def load_thresholded_image(image_path, x=11, y=5):
    data = read_data(image_path).reshape(16 * 660, 512, order="A")
    data *= 255/data.max()
    data = data.astype(np.uint8)
    return cv2.adaptiveThreshold(data,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,x,y)

class ResidueIterator:
    def __init__(self, ids, data_path, pca_model, repeating=True, x=11, y=5, threshold=230,pool_size=16):
        self.ids=ids
        self.data_path=data_path
        self.i = -1
        self.repeating = repeating
        self.pca_model = pca_model
        self.x=x
        self.y=y
        self.threshold=threshold
        self.pool_size=pool_size

    def __iter__(self):
        return self

    def calculate_residue(self, data):
        transformed = self.pca_model.transform(data)
        reverse_transformed = self.pca_model.inverse_transform(transformed)
        reverse_transformed = reverse_transformed[0]
        #skimage.io.imsave("./output_images/" + self.ids[self.i] + "reverse_transformed.png", self.scale_array(reverse_transformed).reshape(-1,512))
        data = data[0]
        binary_transformed = reverse_transformed.copy()
        binary_transformed[reverse_transformed > self.threshold] = 0
        binary_transformed[reverse_transformed < self.threshold] = 1
        binary_data = data.copy()
        binary_data[data > 1] = 0
        binary_data[data < 1] = 1
        diff = binary_data - binary_transformed
        diff[diff<=0] = 0
        diff[diff>0] = 255
        return diff

    def scale_array(self, array):
        mn = np.min(array)
        mx = np.max(array)
        return np.uint8((array - mn)*255/(mx - mn))

    def build_residue(self):
        data = [np.stack(load_thresholded_image(self.data_path + self.ids[self.i] +  ".aps",x=self.x,y=self.y)).flatten()]
        #skimage.io.imsave("./output_images/" + self.ids[self.i] + "original.png", data[0].reshape(-1,512))
        data = self.calculate_residue(data)
        data = skimage.measure.block_reduce(data.reshape(-1, 660, 512, order="A"), (1,self.pool_size,self.pool_size), np.mean)
        #skimage.io.imsave("./output_images/" + self.ids[self.i] + "data.png", self.scale_array(data).reshape(-1,32))
        data = data.astype("float32")
        return data


    def __next__(self):
        if self.i < len(self.ids) -1:
            self.i = self.i + 1
            data = self.build_residue()
            return data
        else:
            if not self.repeating:
                raise StopIteration()
            #Restart iteration, cycle back through
            self.i = -1
            data = self.build_residue()
            return data

class InputImagesIterator:
    def __init__(self, ids, data_path, contrast=1, vertical="both", horizontal="both", repeating=True, pool_size=1, scaler_ids=None, file_format=".aps", seed=111, randomize=False, normalize=True):
        self.ids=ids
        self.contrast = contrast
        self.data_path=data_path
        self.i = -1
        self.vertical = vertical
        self.horizontal = horizontal
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
        print(j)
        print("IMAGE ID " + str(self.ids[j]))
        data = read_data(str(self.data_path + str(self.ids[j]) + str(self.file_format)), self.vertical, self.horizontal, normalize=self.normalize)
        data = skimage.measure.block_reduce(data.reshape(-1, 660, 512, order="A"), (1,self.pool_size,self.pool_size), np.mean).astype(np.float32) * self.contrast
        print("IMG ITER " + str(j))
        
        return data

    def __next__(self):
         # print ("id " + str(self.ids[self.i-1]))
         # print("image iter " + str(self.i))
         if self.i < len(self.ids) -1:
             self.i = self.i + 1
             #print("IMAGES ITERATOR " + str(self.ids[self.i - 1]))
             return self.calculate_data()
         else:
             if not self.repeating:
                 raise StopIteration()
             #Restart iteration, cycle back through
             self.i = 0
             return self.calculate_data()

class SampledImagesIterator:
    def __init__(self, ids, data_path, contrast=1, vertical="both", horizontal="both", repeating=True, pool_size=1, scaler_ids=None, file_format=".aps", seed=111, randomize=False):
        self.ids=ids
        self.contrast = contrast
        self.data_path=data_path
        self.i = -1
        self.vertical = vertical
        self.horizontal = horizontal
        self.repeating = repeating
        self.pool_size = pool_size
        self.file_format = file_format
        self.seed=seed
        self.randomize=randomize
        self.local_random = random.Random()
        self.local_random.seed(seed)

    def __iter__(self):
        return self
    
    def calculate_data(self):
        if self.randomize:
            j = self.local_random.randint(-1,len(self.ids)-1)
        else:
            j = self.i
        print("IMAGE ID " + str(self.ids[j]))
        print("IMAGE ITERATOR " + str(j))
        data = read_data(str(self.data_path + str(self.ids[j]) + str(self.file_format)), self.vertical, self.horizontal, pool_size=self.pool_size)
        return sample_data(data)

    def __next__(self):
         # print ("id " + str(self.ids[self.i-1]))
         # print("image iter " + str(self.i))
         if self.i < len(self.ids) -1:
             self.i = self.i + 1
             #print("IMAGES ITERATOR " + str(self.ids[self.i - 1]))
             return self.calculate_data()
         else:
             if not self.repeating:
                 raise StopIteration()
             #Restart iteration, cycle back through
             self.i = -1
             return self.calculate_data()
        

class GrayscaleImagesIterator:
    def __init__(self, ids, data_path, contrast=1, vertical="both", horizontal="both", repeating=True, pool_size=1, scaler_ids=None, file_format=".aps"):
        self.ids=ids
        self.contrast = contrast
        self.data_path=data_path
        self.i = -1
        self.vertical = vertical
        self.horizontal = horizontal
        self.repeating = repeating
        self.pool_size = pool_size
        self.file_format=file_format

    def __iter__(self):
        return self

    def calculate_data(self):
        data = read_data(self.data_path + self.ids[self.i] + self.file_format, self.vertical, self.horizontal).reshape(-1, 512, order="A")
        data = convert_to_grayscale(data)
        data = spread_spectrum(data)
        data = skimage.measure.block_reduce(data.reshape(-1, 660, 512, order="A"), (1,self.pool_size,self.pool_size), np.mean).astype(np.float32) * self.contrast
        data = normalize(data)
        data = zero_center(data)
        return data

    def __next__(self):
         if self.i < len(self.ids) -1:
             self.i = self.i + 1
             #print("IMAGES ITERATOR " + str(self.ids[self.i - 1]))
             return self.calculate_data()
         else:
             if not self.repeating:
                 raise StopIteration()
             #Restart iteration, cycle back through
             self.i = -1
             return self.calculate_data()

class ScaledImagesIterator:
    def __init__(self, ids, data_path, contrast=1, vertical="both", horizontal="both", repeating=True, pool_size=1, scaler_ids=[], invert=1, file_format=".aps", seed=111, randomize=False, normalize=True):
        self.ids=ids
        self.contrast = contrast
        self.data_path=data_path
        self.i = -1
        self.vertical = vertical
        self.horizontal = horizontal
        self.repeating = repeating
        self.pool_size = pool_size
        self.invert = invert
        self.file_format=file_format
        if len(scaler_ids) != 0:
            self.scaler = self.fit_scaler(scaler_ids)
        else:
            self.scaler = self.fit_scaler(ids)
        self.randomize=randomize
        self.local_random = random.Random()
        self.local_random.seed(seed)
        self.normalize=normalize
        
    def fit_scaler(self,ids):
        scaler = StandardScaler(with_mean=True)
        data = []
        for image in ids[:50]:
            data.append(read_data(str(self.data_path) + "/" + str(image) + self.file_format).flatten())
        data = np.stack(data)
        scaler.fit(data)
        return scaler

    def calculate_data(self):
        if self.randomize:
            j = self.local_random.randint(-1,len(self.ids)-1)
        else:
            j = self.i
        data = read_data(str(self.data_path + str(self.ids[j]) + str(self.file_format)), self.vertical, self.horizontal, normalize=self.normalize)
        data = self.scaler.transform([data.flatten()])
        data = skimage.measure.block_reduce(data.reshape(-1, 660, 512, order="A"), (1,self.pool_size,self.pool_size), np.mean).astype(np.float32) * self.contrast
        return data

    def __next__(self):
         # print ("id " + str(self.ids[self.i-1]))
         # print("image iter " + str(self.i))
         if self.i < len(self.ids) -1:
             self.i = self.i + 1
             #print("IMAGES ITERATOR " + str(self.ids[self.i - 1]))
             return self.calculate_data()
         else:
             if not self.repeating:
                 raise StopIteration()
             #Restart iteration, cycle back through
             self.i = 0
             return self.calculate_data()

    def __iter__(self):
        return self

class ThresholdedInputImagesIterator:
    def __init__(self, ids, data_path, contrast=1, vertical="both", horizontal="both", repeating=True, pool_size=1, scaler_ids=None):
        self.ids=ids
        self.contrast = contrast
        self.data_path=data_path
        self.i = -1
        self.vertical = vertical
        self.horizontal = horizontal
        self.repeating = repeating
        self.pool_size=pool_size

    def __iter__(self):
        return self

    def calculate_data(self):
        data = load_thresholded_image(self.data_path + self.ids[self.i] + ".aps")
        data = skimage.measure.block_reduce(data.reshape(-1, 660, 512, order="A"), (1,self.pool_size,self.pool_size), np.mean).astype(np.float32)
        return data

    def __next__(self):
         if self.i < len(self.ids) -1:
             self.i = self.i + 1
             #print("IMAGES ITERATOR " + str(self.ids[self.i - 1]))
             return(self.calculate_data())
         else:
             if not self.repeating:
                 raise StopIteration()
             #Restart iteration, cycle back through
             self.i = -1
             return(self.calculate_data())


class ThresholdedScaledImagesIterator:
    def __init__(self, ids, data_path, contrast=1, vertical="both", horizontal="both", repeating=True, pool_size=1, scaler_ids=[],x=11,y=5):
        self.ids=ids
        self.contrast = contrast
        self.data_path=data_path
        self.i = -1
        self.vertical = vertical
        self.horizontal = horizontal
        self.repeating = repeating
        self.pool_size=pool_size
        self.x = x
        self.y = y
        if len(scaler_ids) != 0:
            self.scaler = self.fit_scaler(scaler_ids)
        else:
            self.scaler = self.fit_scaler(ids)

    def __iter__(self):
        return self
    
    def fit_scaler(self,ids):
        scaler = StandardScaler()
        data = []
        for image in ids[:300]:
            data.append(load_thresholded_image(str(self.data_path) + "/" + str(image) + ".aps", x=self.x, y=self.y).flatten())
        data = np.stack(data)
        scaler.fit(data)
        return scaler

    def calculate_data(self):
        data = load_thresholded_image(self.data_path + self.ids[self.i] + ".aps", x=self.x, y=self.y)
        data = self.scaler.transform([data.flatten()])
        data = skimage.measure.block_reduce(data.reshape(-1, 660, 512, order="A"), (1,self.pool_size,self.pool_size), np.mean).astype(np.float32)
        return data

    def __next__(self):
         if self.i < len(self.ids) -1:
             self.i = self.i + 1
             #print("IMAGES ITERATOR " + str(self.ids[self.i - 1]))
             return(self.calculate_data())
         else:
             if not self.repeating:
                 raise StopIteration()
             #Restart iteration, cycle back through
             self.i = -1
             return(self.calculate_data())
        
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
        #print("in label iter " + str(self.ids[self.i -1]))
        #print("in label iter " + str(self.test_labels[self.i -1]))
        #print("label iter " + str(self.i))
        
        if self.i < len(self.ids) -1:
            self.i = self.i + 1
            if self.randomize:
                j = self.local_random.randint(-1,len(self.ids)-1)
            else:
                j = self.i
            #print("LABEL " + str(j))
            #print("LABEL" + str(self.test_labels[self.i -1]))
            print("LABEL ITER " + str(j))
            print("LABEL ID " + str(self.labels[j]))
            return(self.labels[j])
        else:
            #Restart iteration, cycle back through
            self.i = -1
            if self.randomize:
                j = self.local_random.randint(-1,len(self.ids)-1)
            else:
                j = self.i
            #raise StopIteration()
            print("LABEL ITER " + str(j))
            print("LABEL ID " + str(self.labels[j]))
            return(self.labels[j])
