import argparse
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

from GAN import GAN

os.makedirs('models/', exist_ok=True)
os.makedirs('summary/', exist_ok=True)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


gan = GAN(do_train=True, input_dim=784, z_dim=20, batch_size=256, scope='GAN')
gan.train_(data_loader=mnist.train, keep_prob=1, weight_decay=0,
    learn_rate_start=0.001, learn_rate_end=0.0001,  n_iter=100000,
    save_model_every_n_iter=99999, path_to_model='models/gan')