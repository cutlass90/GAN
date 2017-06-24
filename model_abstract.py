import os
import time
import itertools as it
import math

import numpy as np
import tensorflow as tf

class Model(object):
    """Abstract class representing a tensorflow scikit-learn-like model.
    
    Class contains several methods used in practically all models."""

    # --------------------------------------------------------------------------
    def save_model(self, path, sess, step=None):
        """ Saves the model to the specified path 
            
            Args:
                path: string path to model to save without extention
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.saver.save(sess, path, global_step=step)
        print("Model saved in file: {}".format(path))
                    

    # --------------------------------------------------------------------------
    def load_model(self, path, sess):
        """ Either load last saved model or specific version

            Args:
                path: string, path to file without extention
                or path to directory(load last saved model)
        """
        load_path = tf.train.latest_checkpoint(path)\
        if os.path.isdir(path) else path
        if load_path is None:
            print('Can not load model, start new train')
            raise FileNotFoundError
        print('try to load {}'.format(load_path))
        self.saver.restore(self.sess, load_path)
        print("Model restored from file %s" % load_path)


    # --------------------------------------------------------------------------
    def create_summary_writers(self, path='summary'):
        os.makedirs(path, exist_ok=True)
        dir_ = os.path.join(path, time.strftime('%d%m%Y_%X'))
        train_writer = tf.summary.FileWriter(logdir=os.path.join(dir_,'train'))
        test_writer = tf.summary.FileWriter(logdir=os.path.join(dir_,'test'))
        return train_writer, test_writer


    # --------------------------------------------------------------------------
    def init_vars(self, vars_, verbose=True):
        self.sess.run(tf.variables_initializer(vars_))
        if verbose:
            print('\nFollowing vars for {} have been initialized:'.format(self.scope))
            for v in vars_:
                print(v.name)


    # --------------------------------------------------------------------------
    def create_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess


    # --------------------------------------------------------------------------
    def number_of_parameters(self, vars_):
        return sum(np.prod(v.get_shape().as_list()) for v in vars_)


    # --------------------------------------------------------------------------
    def scaled_exp_decay(self, start, end, n_iter, current_iter):
        b = math.log(start/end, n_iter)
        a = start*math.pow(1, b)
        learn_rate = a/math.pow((current_iter+1), b)
        return learn_rate

