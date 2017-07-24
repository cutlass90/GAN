import argparse
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

from GAN import GAN


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
gan = GAN(do_train=True, input_dim=784, n_classes=10, z_dim=20, scope='GAN')
gan.load_model(path='models/', sess=gan.sess)
gan.train_(data_loader=mnist, batch_size=128, keep_prob=1, weight_decay=0,
    learn_rate_start=0.001, learn_rate_end=0.0001,  n_iter=100000,
    save_model_every_n_iter=10000, path_to_model='models/gan')
