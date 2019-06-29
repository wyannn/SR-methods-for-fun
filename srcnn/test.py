# -*- coding: utf-8 -*-


import tensorflow as tf
import os
import numpy as np
from PIL import Image


test = 'test'


pic_list = os.listdir(test)

for i in range(len(pic_list)):
    if ('m' in test and os.path.splitext(pic_list[i])[1] == '.bmp'):
        
        image = np.array(Image.open(test + '/' + pic_list[i])).astype(np.float32) / 255.

        input_image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])

        tf.reset_default_graph()
    
        images = tf.placeholder(tf.float32, [1, image.shape[0], image.shape[1], 1], name='images_mos')

        def model(self):

            conv1 = tf.contrib.layers.conv2d(self.images, 64, kernel_size=(9, 9), stride=1, padding='SAME')
            conv2 = tf.contrib.layers.conv2d(conv1, 32, kernel_size=(1, 1), stride=1, padding='SAME')
            out = tf.contrib.layers.conv2d(conv2, 1, kernel_size=(5, 5), stride=1, padding='SAME', activation_fn=None)

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