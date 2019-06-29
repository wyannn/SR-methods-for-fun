# -*- coding: utf-8 -*-

from utils import read_data
import time
import os
import tensorflow as tf
import numpy as np

class VDSR(object):

    def __init__(self, sess, image_size=64, label_size=64, batch_size=64,
                 c_dim=1, checkpoint_dir=None, training=True, scale=3):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.training = training
        self.scale = scale
        self.build_model()

    def build_model(self):
        
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, self.c_dim], name='labels')

        #srcnn
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

        conv1 = tf.contrib.layers.conv2d(self.images, 64, kernel_size=(3,3), stride=1, padding='SAME')
        conv2 = tf.contrib.layers.conv2d(conv1, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv3 = tf.contrib.layers.conv2d(conv2, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv4 = tf.contrib.layers.conv2d(conv3, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv5 = tf.contrib.layers.conv2d(conv4, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv6 = tf.contrib.layers.conv2d(conv5, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv7 = tf.contrib.layers.conv2d(conv6, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv8 = tf.contrib.layers.conv2d(conv7, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv9 = tf.contrib.layers.conv2d(conv8, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv10 = tf.contrib.layers.conv2d(conv9, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv11 = tf.contrib.layers.conv2d(conv10, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv12 = tf.contrib.layers.conv2d(conv11, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv13 = tf.contrib.layers.conv2d(conv12, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv14 = tf.contrib.layers.conv2d(conv13, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv15 = tf.contrib.layers.conv2d(conv14, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv16 = tf.contrib.layers.conv2d(conv15, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv17 = tf.contrib.layers.conv2d(conv16, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        conv18 = tf.contrib.layers.conv2d(conv17, 32, kernel_size=(3, 3), stride=1, padding='SAME')
        conv19 = tf.contrib.layers.conv2d(conv18, 32, kernel_size=(3, 3), stride=1, padding='SAME')
        out = tf.contrib.layers.conv2d(conv19, 1, kernel_size=(3, 3), stride=1, padding='SAME', activation_fn=None)

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