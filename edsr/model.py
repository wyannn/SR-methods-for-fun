# -*- coding: utf-8 -*-

from utils import read_data, resBlock, upsample
import time
import os
import tensorflow as tf
import numpy as np

class EDSR(object):

    def __init__(self, sess, image_size=64, label_size=64, batch_size=64,
                 c_dim=1, checkpoint_dir=None, training=True, scale=3, scaling_factor=0.1, feature_size=256):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.training = training
        self.scale = scale
        self.scaling_factor = scaling_factor
        self.feature_size = feature_size
        self.build_model()

    def build_model(self):
        
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, self.c_dim], name='labels')

        self.pred = self.model()
        self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.labels, self.pred))
        
        #tf.summary.scalar("loss",self.loss)
        #self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def train(self, Config):
        
        data_dir = os.path.join('./{}'.format(Config.checkpoint_dir), Config.data_dir) #获取训练数据的地址
        
        train_data, train_label = read_data(data_dir, Config)

        self.train = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss)

        tf.global_variables_initializer().run()
        
        #summary_writer = tf.summary.FileWriter("./graph",graph=tf.get_default_graph())
        
        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("Load SUCCESS.")
        else:
            print("Load failed!")

        print("Training...")

        for ep in range(Config.epoch):
            batch_idxs = len(train_data) // Config.batch_size
            
            #shuffle
            permutation = np.random.permutation(train_data.shape[0])

            minn = 10000
            for idx in range(0, batch_idxs):
                batch_images = train_data[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                batch_labels = train_label[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]

                counter += 1

                _, err = self.sess.run([self.train, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})


                if counter % 100 == 0:
                    #summary = self.sess.run(self.merged, feed_dict={self.images_g: batch_images_g, self.images_r: batch_images_r, self.images_b: batch_images_b, self.labels_g: batch_labels_g, self.labels_r: batch_labels_r, self.labels_b: batch_labels_b})

                    #summary_writer.add_summary(summary, counter)
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                             % ((ep+1), counter, time.time()-start_time, err))
                
                if counter % 10000 == 0:
                   self.save(Config.checkpoint_dir, counter)
                if err <= minn:
                    minn = err
                    self.save(Config.checkpoint_dir, counter)
            self.save(Config.checkpoint_dir, counter)
                    
    
    def model(self):

        x = tf.contrib.layers.conv2d(self.images, 64, kernel_size=(3,3), stride=1, padding='SAME')
        conv1 = x

        for i in range(64):
            x = resBlock(x, self.feature_size, scale=self.scaling_factor)

        x = tf.contrib.layers.conv2d(x, 64, kernel_size=(3,3), stride=1, padding='SAME')
        x += conv1
        x = upsample(x, self.scale, self.feature_size, None)
        out = x

        return out

    def save(self, checkpoint_dir, step):
        model_name = "TRY.model"
        model_dir = "%s_%s" % ("try", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print("Reading checkpoints...")
        model_dir = "%s_%s" % ("try", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False