# -*- coding: utf-8 -*-

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
 
def read_data(path, Config):

    with h5py.File(path, 'r') as hf:
        data_g = np.array(hf.get('data'))
        label_g = np.array(hf.get('label'))
        data_g = np.reshape(data_g, [data_g.shape[0], Config.image_size, Config.image_size, Config.c_dim])
        label_g = np.reshape(label_g, [label_g.shape[0], Config.label_size, Config.label_size, Config.c_dim])
        return data_g, label_g
            
