# -*- coding: utf-8 -*-


from model import EDSR
import tensorflow as tf
import os

class Config():

    epoch = 20
    learning_rate = 1e-4
    batch_size = 32
    image_size = 32
    label_size = 96
    c_dim = 1
    checkpoint_dir = 'checkpoint'
    data_dir = 'train.h5'
    scale = 3
    feature_size = 256
    scaling_factor = 0.1

def main():

    if not os.path.exists(Config.checkpoint_dir):
        os.makedirs(Config.checkpoint_dir)

    with tf.Session() as sess:
        trysr = EDSR(sess,
                     image_size = Config.image_size,
                     label_size = Config.label_size,
                     batch_size = Config.batch_size,
                     c_dim = Config.c_dim,
                     checkpoint_dir = Config.checkpoint_dir,
                     scale = Config.scale,
                     feature_size = Config.feature_size,
                     scaling_factor = Config.scaling_factor)

        trysr.train(Config)

if __name__ == '__main__':
  main()