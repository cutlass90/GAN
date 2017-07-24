import time
from tqdm import tqdm

import tensorflow as tf
import numpy as np

from classifier import Classifier
from GAN import GAN
from model_abstract.model_abstract import Model

class SemiClassifier(Model):

    # --------------------------------------------------------------------------
    def __init__(self, gan, classifier, do_train):

        self.gan = gan
        self.classifier = classifier
        self.do_train = do_train


    # --------------------------------------------------------------------------
    def train_model(self, labeled_data_loader, labeled_data_loader, batch_size,
        weight_decay, n_iter, learn_rate, keep_prob, save_model_every_n_iter):
