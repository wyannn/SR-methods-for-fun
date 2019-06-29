# -*- coding: utf-8 -*-


import tensorflow as tf
import os
import numpy as np
from PIL import Image
from utils import resBlock, upsample


test = 'test'
scaling_factor = 0.1
feature_size = 256
sacle = 3

pic_list = os.listdir(test)

for i in range(len(pic_list)):
    if ('m' in test and os.path.splitext(pic_list[i])[1] == '.bmp'):
        
        image = np.array(Image.open(test + '/' + pic_list[i])).astype(np.float32) / 255.

        input_image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        tf.reset_default_graph()
    
        images = tf.placeholder(tf.float32, [1, image.shape[0], image.shape[1], 1], name='images_mos')


        def model():

            x = tf.contrib.layers.conv2d(images, 64, kernel_size=(3, 3), stride=1, padding='SAME')
            conv1 = x

            for i in range(64):
                x = resBlock(x, 256, 0.1)

            x = tf.contrib.layers.conv2d(x, 64, kernel_size=(3, 3), stride=1, padding='SAME')
            x += conv1
            x = upsample(x, 3, 256, None)
            out = x

            return out

        output = model()

        checkpoint_dir = 'checkpoint/try_64'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        model_dir = os.path.join(checkpoint_dir, ckpt_name)
        #model_dir = 'checkpoint/try_40/TRY.model-8885'

        saver = tf.train.Saver() 
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver.restore(sess, model_dir)
            results = sess.run(output, feed_dict={images: input_image})

        res = (np.squeeze(results)) * 255.
        Image.fromarray(np.uint8(res)).save('res/' + pic_list[i][:-4] + '.bmp')