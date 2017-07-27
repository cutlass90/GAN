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

        self.disc_sum, self.gen_sum, self.class_sum = [], [], []
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)
            with tf.variable_scope(scope):
                self.create_graph()
            # [print(i) for i in tf.trainable_variables()]
            if do_train:
                self.discriminator_cost = self.get_discriminator_cost(self.logits_critic_r,
                    self.logits_critic_f)
                self.generator_cost = self.get_generator_cost(self.logits_class_f,
                    self.logits_critic_f, self.logits_gen)
                self.classifier_cost = self.get_classifier_cost(self.labels,
                    self.logits_class_r)
                
                self.train_disc, self.train_gen, self.train_class = self.create_optimizer_graph(
                    self.discriminator_cost, self.generator_cost, self.classifier_cost)
                self.train_writer, self.test_writer = self.create_summary_writers('summary/GAN')
                self.disc_merge = tf.summary.merge(self.disc_sum)
                self.gen_merge = tf.summary.merge(self.gen_sum)
                self.class_merge = tf.summary.merge(self.class_sum)

            self.sess = self.create_session()
            # self.train_writer.add_graph(tf.get_default_graph())
            self.sess.run(tf.global_variables_initializer())
            self.stored_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            self.saver = tf.train.Saver(self.stored_vars, max_to_keep=1000)


    # --------------------------------------------------------------------------
    def create_graph(self):
        print('Creat graph')
        self.inputs,\
        self.z,\
        self.labels,\
        self.keep_prob,\
        self.weight_decay,\
        self.learn_rate,\
        self.is_training = self.input_graph()

        self.x_fake, self.logits_gen = self.generator(z=self.z,
            structure=[256, 256, self.input_dim+self.n_classes])
        self.gen_pred = tf.reduce_max(self.logits_gen, axis=1)
        self.gen_pred = tf.cast(tf.equal(self.logits_gen, tf.expand_dims(self.gen_pred, 1)), tf.float32)

        with tf.variable_scope('discriminator'):
            dense_r = self.dense(self.inputs, structure=[256, 256], reuse=False) # b x 1
            dense_f = self.dense(self.x_fake, structure=[256, 256], reuse=True) # b x 1
            self.logits_critic_r = tf.layers.dense(inputs=dense_r, units=1,
                activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=False, name='logits_critic')# b x 1
            self.logits_critic_f = tf.layers.dense(inputs=dense_f, units=1,
                activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=True, name='logits_critic')# b x 1
            self.logits_class_r = tf.layers.dense(inputs=dense_r, units=self.n_classes,
                activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=False, name='logits_class')# b x 10
            self.logits_class_f = tf.layers.dense(inputs=dense_f, units=self.n_classes,
                activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=True, name='logits_class')# b x 10

        # self.logits_critic_r = self.discriminator(self.inputs, structure=[256, 256, 1],
        #     reuse=False) # b x 1
        # self.logits_critic_f = self.discriminator(self.x_fake, structure=[256, 256, 1],
        #     reuse=True) # b x 1

        # self.logits_class_r = self.classifier(self.inputs,
        #     structure=[256, 256, self.n_classes], reuse=False) # b x 10
        # self.logits_class_f = self.classifier(self.x_fake,
        #     structure=[256, 256, self.n_classes], reuse=True) # b x 10

        print('Done!')


    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')
        inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='inputs')
        labels = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='labels')
        z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        weight_decay = tf.placeholder(tf.float32, name='weight_decay')
        learn_rate = tf.placeholder(tf.float32, name='learn_rate')
        is_training = tf.placeholder(tf.bool, name='is_training')
        return inputs, z, labels, keep_prob, weight_decay, learn_rate, is_training


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
        self.gen_sum.append(tf.summary.image('generated img', images, max_outputs=12))
        return x_fake, logits_fake


    # --------------------------------------------------------------------------
    def dense(self, inputs, structure, reuse):
        print('\tdiscriminator')
        for i, layer in enumerate(structure):
            inputs = tf.layers.dense(inputs=inputs, units=layer, activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=reuse, name='dense'+str(i))
        return inputs


    # --------------------------------------------------------------------------
    # def discriminator(self, x, structure, reuse):
    #     print('\tdiscriminator')
    #     with tf.variable_scope('discriminator'):
    #         for i, layer in enumerate(structure[:-1]):
    #             x = tf.layers.dense(inputs=x, units=layer, activation=None,
    #                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
    #                 reuse=reuse, name='disc'+str(i))
    #             # x = tf.contrib.layers.batch_norm(inputs=x, scale=True,
    #             #     updates_collections=None, is_training=self.is_training)
    #             x = tf.nn.elu(x)
    #         x = tf.layers.dense(inputs=x, units=structure[-1], activation=None,
    #             kernel_initializer=tf.contrib.layers.xavier_initializer(),
    #             reuse=reuse, name='disc_last')
    #     return x


    # --------------------------------------------------------------------------
    # def classifier(self, x, structure, reuse):
    #     print('\tclassifier')
    #     with tf.variable_scope('classifier'):
    #         for i, layer in enumerate(structure[:-1]):
    #             x = tf.layers.dense(inputs=x, units=layer, activation=None,
    #                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
    #                 reuse=reuse, name='class'+str(i))
    #             # x = tf.contrib.layers.batch_norm(inputs=x, scale=True,
    #             #     updates_collections=None, is_training=self.is_training)
    #             x = tf.nn.elu(x)
    #         x = tf.layers.dense(inputs=x, units=structure[-1], activation=None,
    #             kernel_initializer=tf.contrib.layers.xavier_initializer(),
    #             reuse=reuse, name='class_last')
    #     return x


    # --------------------------------------------------------------------------
    def get_discriminator_cost(self, logits_critic_r, logits_critic_f):
        print('get_discriminator_cost')
        
        ones = tf.ones(shape=tf.shape(logits_critic_r))
        zeros = tf.zeros(shape=tf.shape(logits_critic_f))

        loss_critic_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=logits_critic_r)) 
        loss_critic_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=logits_critic_f)) 
        cost = loss_critic_r + loss_critic_f

        with tf.name_scope('discriminator'):
            self.disc_sum.append(tf.summary.scalar('loss_critic_r', loss_critic_r))
            self.disc_sum.append(tf.summary.scalar('loss_critic_f', loss_critic_f))
        return cost


    # --------------------------------------------------------------------------
    def get_classifier_cost(self, labels, logits_class_r):
        print('get_classifier_cost')
        loss_class_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits_class_r))  # b x 10

        with tf.name_scope('GANs_classifier'):
            self.class_sum.append(tf.summary.scalar('loss_class_r', loss_class_r))
            precision, recall, f1, accuracy = self.get_metrics(labels, logits_class_r)
            for i in range(self.n_classes):
                self.class_sum.append(tf.summary.scalar('Class {} f1 score'.format(i), f1[i]))
            self.class_sum.append(tf.summary.scalar('Accuracy', accuracy))

        return loss_class_r


    # --------------------------------------------------------------------------
    def get_generator_cost(self, logits_class_f, logits_critic_f,
        logits_gen):
        """
            Args:
                logits_class_f: tensor, logits from class on fake data
                logits_critic_f: tensor, logits from critic about fake data
                logits_gen: tensor, logits from generator on it's own fake data
        """
        print('get_generator_cost')

        # classification loss
        labels = tf.nn.softmax(logits_class_f/20)
        loss_class_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits_gen))

        # Adversarial loss
        ones = tf.ones(shape=tf.shape(logits_critic_f))
        loss_critic_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=logits_critic_f))

        # Distribution loss
        class_distrib = tf.reduce_mean(tf.nn.softmax(logits_gen/0.1),0) # 10
        # class_distrib = tf.Print(class_distrib, [labels[0,:]], summarize=100)
        target_distrib=tf.constant(value=1./self.n_classes, shape=[self.n_classes],
                dtype=tf.float32)
        loss_distrib = -tf.reduce_sum(target_distrib*tf.log(class_distrib+1e-6))
            
        cost = loss_critic_f + loss_class_f + loss_distrib

        #summary
        class_balance_f = tf.argmax(logits_gen, axis=1)
        class_balance_r = tf.argmax(logits_class_f, axis=1)
        self.gen_sum.append(tf.summary.histogram('class_balance_generator', class_balance_f))
        self.gen_sum.append(tf.summary.histogram('class_balance_critic', class_balance_r))
        with tf.name_scope('generator'):
            self.gen_sum.append(tf.summary.scalar('loss_class_f', loss_class_f))
            self.gen_sum.append(tf.summary.scalar('loss_critic_f', loss_critic_f))
            self.gen_sum.append(tf.summary.scalar('loss_distrib', loss_distrib))
            labels = tf.reduce_max(logits_class_f, axis=1)
            labels = tf.cast(tf.equal(logits_class_f, tf.expand_dims(labels, 1)), tf.float32)
            precision, recall, f1, accuracy = self.get_metrics(labels, logits_gen)
            for i in range(self.n_classes):
                self.gen_sum.append(tf.summary.scalar('Class {} f1 score'.format(i), f1[i]))
            self.gen_sum.append(tf.summary.scalar('Accuracy', accuracy))
        return cost


    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, disc_cost, gen_cost, class_cost):
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

            class_optimizer = tf.train.AdamOptimizer(self.learn_rate)
            train_class = class_optimizer.minimize(class_cost,
                var_list=tf.get_collection('trainable_variables',
                    scope=self.scope+'/discriminator'))
        return train_disc, train_gen, train_class


    #---------------------------------------------------------------------------
    def train_(self, data_loader, batch_size, keep_prob, weight_decay,  learn_rate_start,
        learn_rate_end, n_iter, save_model_every_n_iter, path_to_model):
        print('\n\n\n\t----==== Training ====----')
            
        start_time = time.time()
        for current_iter in tqdm(range(n_iter)):
            learn_rate = self.scaled_exp_decay(learn_rate_start, learn_rate_end,
                n_iter, current_iter)
            batch = data_loader.train.next_batch(batch_size)
            z = np.random.normal(size=[batch_size, self.z_dim])
            feedDict = {self.inputs : batch[0],
                        self.z :z,
                        self.labels : batch[1],
                        self.keep_prob : keep_prob,
                        self.weight_decay : weight_decay,
                        self.learn_rate : learn_rate,
                        self.is_training : True}
                        
            self.sess.run(self.train_disc, feed_dict=feedDict)
            self.sess.run(self.train_gen, feed_dict=feedDict)
            self.sess.run(self.train_class, feed_dict=feedDict)
            if current_iter%200 == 0:
                summary = self.sess.run(self.disc_merge, feed_dict=feedDict)
                self.train_writer.add_summary(summary, current_iter)
                summary = self.sess.run(self.gen_merge, feed_dict=feedDict)
                self.train_writer.add_summary(summary, current_iter)
                summary = self.sess.run(self.class_merge, feed_dict=feedDict)
                self.train_writer.add_summary(summary, current_iter)

                batch = data_loader.test.next_batch(batch_size)
                feedDict[self.is_training]=False
                feedDict[self.inputs]=batch[0]
                feedDict[self.labels]=batch[1]
                summary = self.sess.run(self.disc_merge, feed_dict=feedDict)
                self.test_writer.add_summary(summary, current_iter)
                summary = self.sess.run(self.gen_merge, feed_dict=feedDict)
                self.test_writer.add_summary(summary, current_iter)
                summary = self.sess.run(self.class_merge, feed_dict=feedDict)
                self.test_writer.add_summary(summary, current_iter)

            if (current_iter+1)%1000 == 0:
                samples = self.sample()
                plot_samples(samples, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)

        self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        print('\nTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))


    # --------------------------------------------------------------------------
    def train_gen_step(self, keep_prob, weight_decay, batch_size, learn_rate):
        z = np.random.normal(size=[batch_size, self.z_dim])
        feedDict = {self.z :z,
                    self.keep_prob : keep_prob,
                    self.weight_decay : weight_decay,
                    self.learn_rate : learn_rate,
                    self.is_training : True}
        self.sess.run(self.train_gen, feed_dict=feedDict)


    # --------------------------------------------------------------------------
    def train_disc_step(self, inputs, keep_prob, weight_decay, learn_rate):
        batch_size = len(inputs)
        z = np.random.normal(size=[batch_size, self.z_dim])
        feedDict = {self.inputs : inputs,
                    self.z : z,
                    self.keep_prob : keep_prob,
                    self.weight_decay : weight_decay,
                    self.learn_rate : learn_rate,
                    self.is_training : True}
        self.sess.run(self.train_disc, feed_dict=feedDict)


    # --------------------------------------------------------------------------
    def train_class_step(self, inputs, labels, keep_prob, weight_decay, learn_rate):
        batch_size = len(inputs)
        feedDict = {self.inputs : inputs,
                    self.labels : labels,
                    self.keep_prob : keep_prob,
                    self.weight_decay : weight_decay,
                    self.learn_rate : learn_rate,
                    self.is_training : True}
        self.sess.run(self.train_class, feed_dict=feedDict)


    # --------------------------------------------------------------------------
    def save_gen_summaries(self, keep_prob, weight_decay, batch_size, is_training,
        writer, it):
        z = np.random.normal(size=[batch_size, self.z_dim])
        feedDict = {self.z :z,
                    self.keep_prob : keep_prob,
                    self.weight_decay : weight_decay,
                    self.is_training : is_training}
        summary = self.sess.run(self.gen_merge, feed_dict=feedDict)
        writer.add_summary(summary, it)


    # --------------------------------------------------------------------------
    def save_disc_summaries(self, inputs, keep_prob, weight_decay, is_training,
        writer, it):
        batch_size = len(inputs)
        z = np.random.normal(size=[batch_size, self.z_dim])
        feedDict = {self.inputs : inputs,
                    self.z : z,
                    self.keep_prob : keep_prob,
                    self.weight_decay : weight_decay,
                    self.is_training : is_training}
        summary = self.sess.run(self.disc_merge, feed_dict=feedDict)
        writer.add_summary(summary, it)


    # --------------------------------------------------------------------------
    def save_class_summaries(self, inputs, labels, keep_prob, weight_decay, is_training,
        writer, it):
        batch_size = len(inputs)
        feedDict = {self.inputs : inputs,
                    self.labels : labels,
                    self.keep_prob : keep_prob,
                    self.weight_decay : weight_decay,
                    self.is_training : is_training}
        summary = self.sess.run(self.class_merge, feed_dict=feedDict)
        writer.add_summary(summary, it)


    # --------------------------------------------------------------------------
    def train_model(self, data_loader, batch_size, keep_prob, weight_decay,  learn_rate_start,
        learn_rate_end, n_iter, save_model_every_n_iter, path_to_model):

        start_time = time.time()
        for current_iter in tqdm(range(n_iter)):
            learn_rate = self.scaled_exp_decay(learn_rate_start, learn_rate_end,
                n_iter, current_iter)
            batch = data_loader.train.next_batch(batch_size)
            self.train_disc_step(batch[0], keep_prob, weight_decay, learn_rate)
            self.train_gen_step(keep_prob, weight_decay, batch_size, learn_rate)
            self.train_class_step(batch[0], batch[1], keep_prob, weight_decay, learn_rate)

            if current_iter%200 == 0:
                self.save_disc_summaries(batch[0], keep_prob, weight_decay, True,
                    self.train_writer, current_iter)
                self.save_gen_summaries(keep_prob, weight_decay, batch_size, True,
                    self.train_writer, current_iter)
                self.save_class_summaries(batch[0], batch[1], keep_prob, weight_decay,
                    True, self.train_writer, current_iter)

                batch = data_loader.test.next_batch(batch_size)
                self.save_disc_summaries(batch[0], keep_prob, weight_decay, False,
                    self.test_writer, current_iter)
                self.save_gen_summaries(keep_prob, weight_decay, batch_size, False,
                    self.test_writer, current_iter)
                self.save_class_summaries(batch[0], batch[1], keep_prob, weight_decay,
                    False, self.test_writer, current_iter)

            if (current_iter+1)%5000 == 0:
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
    

    #---------------------------------------------------------------------------
    def sample_data(self, batch_size):
        z = np.random.normal(size=[batch_size, self.z_dim])
        images, labels = self.sess.run([self.x_fake, self.gen_pred], {self.is_training:False, self.z:z})
        return images, labels
