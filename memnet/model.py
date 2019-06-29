# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:54:09 2018

@author: student
"""
from utils import read_data, memnet_m6r6
import time
import os
import tensorflow as tf
import numpy as np

class MEMNET(object):

    def __init__(self, sess, image_size = 41, label_size = 41, batch_size = 64,
                 c_dim = 1, checkpoint_dir = None, training = True, scale=3):

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
        
        

        #mae
        self.pred, self.loss = memnet_m6r6(name="MemNet_M6R6", clean_data=self.labels, noisy_data=self.images, num_filters=64, image_c=self.c_dim, is_training=True, reuse=False) 
        
       # MemNet_M6R6 Variables
        self.memnet_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="MemNet_M6R6")

        # optim, SGD.
        '''
        Optim methods have:
        tf.train.AdadeltaOptimizer, tf.train.AdagradDAOptimizer, tf.train.AdagradOptimizer, tf.train.AdamOptimizer
        tf.train.MomentumOptimizer, tf.train.RMSPropOptimizer and so on.
        '''
        # summary. only add loss. 
        tf.summary.scalar("loss",self.loss)
        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def train(self, Config):
        
        data_dir = os.path.join('./{}'.format(Config.checkpoint_dir), Config.data_dir) #获取训练数据的地址

        train_data, train_label = read_data(data_dir, Config)        
        
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss)
        """
        
        #self.train_op = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(Config.learning_rate, global_step*Config.batch_size, 40*len(train_data)*Config.batch_size, 0.5, staircase=True)
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss, var_list=self.memnet_variables, name="memnet_opt")
        
        #gradient clip
        """
        global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(Config.learning_rate, global_step*Config.batch_size, 5*len(train_data)*Config.batch_size, 0.1, staircase=True)        
        
        opt = tf.train.AdamOptimizer(learning_rate=Config.learning_rate)
        grad_and_value = opt.compute_gradients(self.loss)
        
        clip = tf.Variable(Config.clip_grad, name='clip') 
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -clip, clip)
        capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in grad_and_value]
        
        self.train_op = opt.apply_gradients(capped_gvs, global_step=global_step)
        """
        
        
        tf.global_variables_initializer().run()
        
        summary_writer = tf.summary.FileWriter("./graph",graph=tf.get_default_graph())
        
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
            #train_data = train_data[permutation,:, :, :]
            #train_label = train_label[permutation,:, :, :]
            
            for idx in range(0, batch_idxs):
                batch_images = train_data[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                batch_labels = train_label[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                
                #permutation = np.random.choice(train_data.shape[0], Config.batch_size)
                #batch_images = train_data[permutation,:, :, :]
                #batch_labels = train_label[permutation,:, :, :]

                counter += 1
                _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
                
                if counter % 100 == 0:
                    summary = self.sess.run(self.merged, feed_dict={self.images: batch_images, self.labels: batch_labels})
                    summary_writer.add_summary(summary, counter)
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                             % ((ep+1), counter, time.time()-start_time, err))
                
                #if counter % 1000 == 0:
                #   self.save(Config.checkpoint_dir, counter)
                #if err < 0.015:
                #    self.save(Config.checkpoint_dir, counter)
            self.save(Config.checkpoint_dir, counter)
            #if ep == 9 or ep == 19 or ep == 29 or ep == 39 or ep == 49 or ep == 59 or ep == 69 or ep == 79:
            #    self.save(Config.checkpoint_dir, counter)
        
                

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