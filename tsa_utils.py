import os 
import numpy as np
import pickle

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


def read_data(infile, vertical="both", horizontal="both"):
    """Read any of the 4 types of image files, returns a numpy array of the image contents
    """
    extension = os.path.splitext(infile)[1]
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512) #skip header
    if extension == '.aps' or extension == '.a3daps':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
        if vertical == "bottom":
            data = data[:, :340, :] 
        elif vertical == "top":
            data = data[:, 320:660, :] 
        elif vertical == "middle":
            data = data[:, 170:510, :] 
        rotated_data = []
        if horizontal == "right":
            for i in range(0,16):
                increment = 10
                rotated_data.append(data[(i * increment) + 50:(i * increment) + 320,:,i])
        elif horizontal == "left":
            for i in range(0,16):
                increment = 10
                rotated_data.append(data[210 - (i * increment): 480-(i * increment):,:,i])
        elif horizontal == "middle":
            for i in range(0,16):
                increment = 10
                rotated_data.append(data[130:400,:,i])
        else:
            for i in range(0,16):
                rotated_data.append(data[:,:,i])
        data = np.array(rotated_data)  
    elif extension == '.a3d':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, nt, ny, order='F').copy() #make N-d image
    elif extension == '.ahi':
        data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
        data = data.reshape(2, ny, nx, nt, order='F').copy()
        real = data[0,:,:,:].copy()
        imag = data[1,:,:,:].copy()
    fid.close()
    if extension != '.ahi':
        return np.swapaxes(data, 2, 1)
    else:
        return real, imag

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

class ResidueIterator:
    def __init__(self, ids, data_path, pca_model, repeating=True):
        self.ids=ids
        self.data_path=data_path
        self.i = -1
        self.repeating = repeating
        self.pca_model = pca_model

    def __iter__(self):
        return self

    def next(self):
        if self.i < len(self.ids) -1:
            self.i = self.i + 1
            print ("RETURNING RESIDUE")
            print(np.stack(read_data(self.data_path + self.ids[self.i] + ".aps", "both", "both")).shape)
            return np.stack(read_data(self.data_path + self.ids[self.i] + ".aps", "both", "both"))
        else:
            if not self.repeating:
                raise StopIteration()
            #Restart iteration, cycle back through
            self.i = -1
            return np.stack(read_data(self.data_path + self.ids[self.i] + ".aps", "both", "both"))

class InputImagesIterator:
    def __init__(self, ids, data_path, contrast=1, vertical="both", horizontal="both", repeating=True):
        self.ids=ids
        self.contrast = contrast
        self.data_path=data_path
        self.i = -1
        self.vertical = vertical
        self.horizontal = horizontal
        self.repeating = repeating

    def __iter__(self):
        return self

    def calculate_residue(self, data):
        transformed = self.pca_model.transform(data)
        reverse_transformed = self.pca_model.inverse_transform(transformed) 
        reverse_transformed = reverse_transformed[0]
        data = data[0]
        binary_transformed = reverse_transformed.copy()
        binary_transformed[reverse_transformed > 230] = 0
        binary_transformed[reverse_transformed < 230] = 1
        binary_data = data.copy()
        binary_data[data > 1] = 0
        binary_data[data < 1] = 1
        diff = binary_data - binary_transformed
        diff[diff==0] = 0
        diff[diff>0] = 255

    def next(self):
        # print ("id " + str(self.ids[self.i-1])) 
        # print("image iter " + str(self.i))
        if self.i < len(self.ids) -1:
            self.i = self.i + 1
            #print("IMAGES ITERATOR " + str(self.ids[self.i - 1]))
            return self.calculate_residue(np.stack(read_data(self.data_path + self.ids[self.i] + ".aps", sel+f.vertical, self.horizontal)))
        else:
            if not self.repeating:
                raise StopIteration()
            #Restart iteration, cycle back through
            self.i = -1
            return self.calculate_residue(np.stack(read_data(self.data_path + self.ids[0] + ".aps", self.vertical, self.horizontal)))

class InputLabelsIterator:
    def __init__(self, ids, labels):
        self.ids=ids
        self.labels=labels
        self.i=-1

    def __iter__(self):
        return self

    def next(self):
        #print("in label iter " + str(self.ids[self.i -1])) 
        #print("in label iter " + str(self.test_labels[self.i -1])) 
        #print("label iter " + str(self.i))
        if self.i < len(self.ids) -1:
            self.i = self.i + 1
            #print("LABEL" + str(self.test_labels[self.i -1]))
            return(self.labels[self.i])
        else:
            #Restart iteration, cycle back through
            self.i = -1
            #raise StopIteration()
            return(self.labels[0])
