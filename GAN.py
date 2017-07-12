import os
import time
import math
import itertools as it

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from model_abstract import Model
from plot import sample as plot_samples

class GAN(Model):

    def __init__(self, do_train, input_dim, n_classes, z_dim, scope):

        self.do_train = do_train
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.z_dim = z_dim
        self.scope = scope

        self.disc_sum, self.gen_sum = [], []
        with tf.variable_scope(scope):
            self.create_graph()
        # [print(i) for i in tf.trainable_variables()]
        if do_train:
            self.discriminator_cost = self.get_discriminator_cost(self.targets,
                self.logits_class_r, self.logits_critic_r, self.logits_critic_f)
            self.generator_cost = self.get_generator_cost(self.logits_class_f,
                self.logits_critic_f, self.logits_fake)
            
            self.train_disc, self.train_gen = self.create_optimizer_graph(
                self.discriminator_cost, self.generator_cost)
            self.train_writer, self.test_writer = self.create_summary_writers()
            self.disc_merge = tf.summary.merge(self.disc_sum)
            self.gen_merge = tf.summary.merge(self.gen_sum)

        self.sess = self.create_session()
        self.train_writer.add_graph(tf.get_default_graph())
        self.sess.run(tf.global_variables_initializer())
        self.stored_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        self.saver = tf.train.Saver(self.stored_vars, max_to_keep=1000)


    # --------------------------------------------------------------------------
    def create_graph(self):
        print('Creat graph')
        self.inputs,\
        self.z,\
        self.targets,\
        self.keep_prob,\
        self.weight_decay,\
        self.learn_rate,\
        self.is_training = self.input_graph()

        self.x_fake, self.logits_fake = self.generator(z=self.z,
            structure=[256, 256, self.input_dim+self.n_classes])

        x = tf.concat((self.inputs, self.x_fake), axis=0)

        logits_class, logits_critic = self.discriminator(x,
            structure=[256, 256, self.n_classes+1]) # b x 10, b x 1

        self.logits_class_r, self.logits_class_f = tf.split(logits_class,
            num_or_size_splits=2, axis=0)
        self.logits_critic_r, self.logits_critic_f = tf.split(logits_critic,
            num_or_size_splits=2, axis=0)

        print('Done!')


    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')
        inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='inputs')
        targets = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='targets')
        z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        weight_decay = tf.placeholder(tf.float32, name='weight_decay')
        learn_rate = tf.placeholder(tf.float32, name='learn_rate')
        is_training = tf.placeholder(tf.bool, name='is_training')
        return inputs, z, targets, keep_prob, weight_decay, learn_rate, is_training


    # --------------------------------------------------------------------------
    def generator(self, z, structure):
        print('\tgenerator')
        with tf.variable_scope('generator'):
            for layer in structure[:-1]:
                z = tf.layers.dense(inputs=z, units=layer, activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
                z = tf.contrib.layers.batch_norm(inputs=z, scale=True,
                    updates_collections=None, is_training=self.is_training)
                z = tf.nn.elu(z)
            z = tf.layers.dense(inputs=z, units=structure[-1], activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            x_fake = tf.sigmoid(z[:, :self.input_dim])
            logits_fake = z[:, self.input_dim:]
        images = tf.reshape(x_fake, [-1, 28, 28, 1])
        self.gen_sum.append(tf.summary.image('generated img', images, max_outputs=100))
        return x_fake, logits_fake


    # --------------------------------------------------------------------------
    def discriminator(self, x, structure):
        print('\tdiscriminator')
        with tf.variable_scope('discriminator'):
            for i, layer in enumerate(structure[:-1]):
                x = tf.layers.dense(inputs=x, units=layer, activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.contrib.layers.batch_norm(inputs=x, scale=True,
                    updates_collections=None, is_training=self.is_training)
                x = tf.nn.elu(x)
            x = tf.layers.dense(inputs=x, units=structure[-1], activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            logits_class = x[:, :self.n_classes] # b x n_classes
            logits_critic = x[:, self.n_classes:] # b x 1 probability of true sample
        return logits_class, logits_critic


    # --------------------------------------------------------------------------
    def get_discriminator_cost(self, targets, logits_class_r, logits_critic_r,
        logits_critic_f):
        print('get_discriminator_cost')
        
        ones = tf.ones(shape=tf.shape(logits_critic_r))
        zeros = tf.zeros(shape=tf.shape(logits_critic_f))

        loss_class_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=targets, logits=logits_class_r))  # b x 10
        loss_critic_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=logits_critic_r)) 
        loss_critic_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=logits_critic_f)) 
        cost = loss_critic_r + loss_critic_f + loss_class_r
        with tf.name_scope('discriminator'):
            self.disc_sum.append(tf.summary.scalar('loss_class_r', loss_class_r))
            self.disc_sum.append(tf.summary.scalar('loss_critic_r', loss_critic_r))
            self.disc_sum.append(tf.summary.scalar('loss_critic_f', loss_critic_f))

            pred = tf.reduce_max(logits_class_r, axis=1)
            pred = tf.cast(tf.equal(logits_class_r, tf.expand_dims(pred, 1)), tf.float32)
            for i in range(self.n_classes):
                y = targets[:,i]
                y_ = pred[:,i]
                tp = tf.reduce_sum(y*y_)
                tn = tf.reduce_sum((1-y)*(1-y_))
                fp = tf.reduce_sum((1-y)*y_)
                fn = tf.reduce_sum(y*(1-y_))
                pr = tp/(tp+fp+1e-5)
                re = tp/(tp+fn+1e-5)
                f1 = 2*pr*re/(pr+re+1e-5)
                with tf.name_scope('Class_{}'.format(i)):
                    self.disc_sum.append(tf.summary.scalar('Class {} precision'.format(i), pr))
                    self.disc_sum.append(tf.summary.scalar('Class {} recall'.format(i), re))
                    self.disc_sum.append(tf.summary.scalar('Class {} f1 score'.format(i), f1))
        return cost


    # --------------------------------------------------------------------------
    def get_generator_cost(self, logits_class_critic, logits_critic_f,
        logits_class_gen):
        """
            Args:
                logits_class_critic: tensor, class logits from critic on fake data,
                that has been provided by generator
                logits_critic_f: tensor, logits from critic about fake data
                logits_class_gen: tensor, class logits from generator
        """
        print('get_generator_cost')

        labels = tf.nn.softmax(logits_class_critic)
        loss_class_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits_class_gen))

        ones = tf.ones(shape=tf.shape(logits_critic_f))
        loss_critic_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=logits_critic_f))

        class_distrib = tf.reduce_mean(tf.nn.softmax(logits_class_critic),0) # 10
        # class_distrib = tf.Print(class_distrib, [class_distrib], summarize=100)
        target_distrib=tf.constant(value=1./self.n_classes, shape=[self.n_classes],
                dtype=tf.float32)
        loss_distrib = -tf.reduce_sum(target_distrib*tf.log(class_distrib+1e-6))
            
        cost = loss_critic_f + loss_class_f + loss_distrib

        class_balance_f = tf.argmax(logits_class_gen, axis=1)
        class_balance_r = tf.argmax(logits_class_critic, axis=1)
        self.gen_sum.append(tf.summary.histogram('class_balance_generator', class_balance_f))
        self.gen_sum.append(tf.summary.histogram('class_balance_critic', class_balance_r))
        with tf.name_scope('generator'):
            self.gen_sum.append(tf.summary.scalar('loss_class_f', loss_class_f))
            self.gen_sum.append(tf.summary.scalar('loss_critic_f', loss_critic_f))
            self.gen_sum.append(tf.summary.scalar('loss_distrib', loss_distrib))
        return cost


    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, disc_cost, gen_cost):
        print('create_optimizer_graph')
        with tf.variable_scope('optimizer_graph'):
            disc_optimizer = tf.train.AdamOptimizer(self.learn_rate)
            disc_list_grad = disc_optimizer.compute_gradients(disc_cost, 
                var_list=tf.get_collection('trainable_variables',
                    scope=self.scope+'/discriminator'))
            disc_grad = tf.reduce_mean([tf.reduce_mean(tf.abs(t))\
                for t,n in disc_list_grad if t is not None])
            self.disc_sum.append(tf.summary.scalar('disc_grad', disc_grad))
            train_disc = disc_optimizer.apply_gradients(disc_list_grad)

            gen_optimizer = tf.train.AdamOptimizer(self.learn_rate)
            gen_list_grad = gen_optimizer.compute_gradients(gen_cost, 
                var_list=tf.get_collection('trainable_variables',
                    scope=self.scope+'/generator'))
            gen_grad = tf.reduce_mean([tf.reduce_mean(tf.abs(t))\
                for t,n in gen_list_grad if t is not None])
            self.gen_sum.append(tf.summary.scalar('gen_grad', gen_grad))
            train_gen = gen_optimizer.apply_gradients(gen_list_grad)
        return train_disc, train_gen


    #---------------------------------------------------------------------------
    def train_(self, data_loader, batch_size, keep_prob, weight_decay,  learn_rate_start,
        learn_rate_end, n_iter, save_model_every_n_iter, path_to_model):
        print('\n\n\n\t----==== Training ====----')
            
        start_time = time.time()
        for current_iter in tqdm(range(n_iter)):
            learn_rate = self.scaled_exp_decay(learn_rate_start, learn_rate_end,
                n_iter, current_iter)
            batch = data_loader.next_batch(batch_size)
            z = np.random.normal(size=[batch_size, self.z_dim])
            feedDict = {self.inputs : batch[0],
                        self.z :z,
                        self.targets : batch[1],
                        self.keep_prob : keep_prob,
                        self.weight_decay : weight_decay,
                        self.learn_rate : learn_rate,
                        self.is_training : True}
                        
            _, summary = self.sess.run([self.train_disc, self.disc_merge],
                feed_dict=feedDict)
            if current_iter%50 == 0:
                self.train_writer.add_summary(summary, current_iter)

            _, summary = self.sess.run([self.train_gen, self.gen_merge],
                feed_dict=feedDict)
            if current_iter%50 == 0:
                self.train_writer.add_summary(summary, current_iter)

            if (current_iter+1)%1000 == 0:
                samples = self.sample()
                plot_samples(samples, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)

        self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        print('\nTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))


    #---------------------------------------------------------------------------
    def sample(self):
        z = np.random.normal(size=[100, self.z_dim])
        samples = self.sess.run(self.x_fake, {self.is_training:False, self.z:z})
        return samples
        