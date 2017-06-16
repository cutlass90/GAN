import os
import time
import math
import itertools as it

import tensorflow as tf
import numpy as np
from tqdm import tqdm





class GAN(object):

    def __init__(self, do_train, input_dim, z_dim, batch_size):

        self.do_train = do_train
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        if batch_size%2 != 0:
            raise ValueError('batch_size must be even')
        self.create_graph()
        os.makedirs('summary', exist_ok=True)
        sub_d = len(os.listdir('summary'))
        self.train_writer = tf.summary.FileWriter(logdir = 'summary/'+str(sub_d))
        self.merged = tf.summary.merge_all()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(var_list=tf.global_variables(),
                                    max_to_keep = 1000)
        
    # --------------------------------------------------------------------------
    def __enter__(self):
        return self


    # --------------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.reset_default_graph()
        if self.sess is not None:
            self.sess.close()
        

    # --------------------------------------------------------------------------
    def create_graph(self):
        print('Creat graph')

        self.inputs,\
        self.keep_prob,\
        self.weight_decay,\
        self.learn_rate = self.input_graph()

        images = tf.reshape(self.inputs, [self.batch_size//2, 28, 28, 1])
        tf.summary.image('original_img', images)
        
        z = tf.random_normal([self.batch_size//2, self.z_dim])
        self.fake_x = self.generator(data_dim=self.input_dim, z=z)

        x = tf.concat((self.inputs, self.fake_x), 0)
        logits = self.discriminator(x) # b x 1

        self.discriminator_cost = self.get_discriminator_cost(logits)
        self.generator_cost = self.get_generator_cost(logits)
        
        self.train_disc, self.train_gen = self.create_optimizer_graph(
            self.discriminator_cost, self.generator_cost)
        
        print('Done!')


    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')
        inputs = tf.placeholder(tf.float32, shape=[self.batch_size//2, self.input_dim], name='inputs')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        weight_decay = tf.placeholder(tf.float32, name='weight_decay')
        learn_rate = tf.placeholder(tf.float32, name='learn_rate')
        return inputs, keep_prob, weight_decay, learn_rate


    # --------------------------------------------------------------------------
    def generator(self, data_dim, z):
        print('\tgenerator')
        with tf.variable_scope('generator'):
            fc = tf.layers.dense(inputs=z, units=data_dim, activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            fc = tf.layers.dense(inputs=fc, units=data_dim, activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer())

        images = tf.reshape(fc, [self.batch_size//2, 28, 28, 1])
        tf.summary.image('generated img', images)
        return fc


    # --------------------------------------------------------------------------
    def discriminator(self, x):
        print('\tdiscriminator')
        with tf.variable_scope('discriminator'):
            fc = tf.layers.dense(inputs=x, units=self.input_dim, activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            fc = tf.layers.dense(inputs=fc, units=1, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        return fc


    # --------------------------------------------------------------------------
    def get_discriminator_cost(self, logits):
        print('\tget_discriminator_cost')
        ones = tf.constant(value=1., shape=[self.batch_size//2,1])
        zeros = tf.constant(value=0., shape=[self.batch_size//2,1])
        true_labels = tf.concat([ones, zeros], axis=0)

        cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_labels,
            logits=logits)

        cost = tf.reduce_mean(cost)
        # tf.summary.scalar('discriminator cost', cost)
        return cost


    # --------------------------------------------------------------------------
    def get_generator_cost(self, logits):
        print('\tget_generator_cost')
        true_labels = tf.constant(value=1., shape=[self.batch_size//2,1])

        cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_labels,
            logits=logits[self.batch_size//2:,:])

        cost = tf.reduce_mean(cost)
        # tf.summary.scalar('generator cost', cost)
        return cost


    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, disc_cost, gen_cost):
        print('create_optimizer_graph')
        [print(i) for i in tf.trainable_variables()]
        with tf.variable_scope('optimizer_graph'):
            disc_optimizer = tf.train.AdamOptimizer(self.learn_rate)
            gen_optimizer = tf.train.AdamOptimizer(self.learn_rate)
            
            train_disc = disc_optimizer.minimize(disc_cost,
                var_list=tf.get_collection('trainable_variables', scope='discriminator'))
            train_gen = gen_optimizer.minimize(gen_cost,
                var_list=tf.get_collection('trainable_variables', scope='generator'))
        return train_disc, train_gen


    #---------------------------------------------------------------------------
    def train_(self, data_loader,  keep_prob, weight_decay,  learn_rate_start,
        learn_rate_end, n_iter, save_model_every_n_iter, path_to_model):
        print('\n\n\n\t----==== Training ====----')
        #try to load model
        try:
            self.load_model(os.path.dirname(path_to_model))
        except:
            print('Can not load model {0}, starting new train'.format(path_to_model))
            
        start_time = time.time()
        b = math.log(learn_rate_start/learn_rate_end, n_iter) 
        a = learn_rate_start*math.pow(1, b)
        for current_iter in tqdm(range(n_iter)):
            learn_rate = a/math.pow((current_iter+1), b)
            batch = data_loader.next_batch(self.batch_size//2)
            feedDict = {self.inputs : batch[0],
                        self.keep_prob : keep_prob,
                        self.weight_decay : weight_decay,
                        self.learn_rate : learn_rate}
                        
            _, summary = self.sess.run([self.train_disc, self.merged],
                feed_dict=feedDict)
            self.train_writer.add_summary(summary, current_iter)

            _, summary = self.sess.run([self.train_gen, self.merged],
                feed_dict=feedDict)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path = path_to_model, step = current_iter+1)

        self.save_model(path = path_to_model, step = current_iter+1)
        print('\nTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))








# testing #####################################################################################################################
if __name__ == '__main__':
    gan = GAN(do_train=True, input_dim=784, z_dim=20, batch_size=256)