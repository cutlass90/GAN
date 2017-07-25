
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from GAN import GAN
from classifier import Classifier
from semi_classifier import SemiClassifier

class ImageProvider:
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        print('total number of images', self.mnist.train.num_examples)
    def next_batch(self, batch_size):
        return self.mnist.train.next_batch(batch_size)[0]


labeled_size = 128
batch_size = 128
weight_decay = 2e-2
n_iter = 500000
learn_rate_start = 1e-3
learn_rate_end = 1e-4
keep_prob = 1
save_model_every_n_iter = 35000
path_to_model_gan = 'models/gan'
path_to_model_ext_class = 'models/class'



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True,
    validation_size=labeled_size)
labeled_data_loader = mnist.validation
print('total number of labeled data', labeled_data_loader.num_examples)
l = labeled_data_loader.labels
print('Distribution of labeled data:')
[print('class '+str(i)) for i in np.sum(l,0)]

test_data_loader = mnist.test
print('total number of test data', test_data_loader.num_examples)
image_provider = ImageProvider()

ext_classifier = Classifier(input_dim=784, n_classes=10, do_train=True,
    scope='ext_classifier')
gan = GAN(do_train=True, input_dim=784, n_classes=10, z_dim=20, scope='GAN')
semi_classifier = SemiClassifier(gan, ext_classifier, do_train=True)
semi_classifier.train_model(image_provider, labeled_data_loader, test_data_loader,
    batch_size, weight_decay, n_iter, learn_rate_start, learn_rate_end, keep_prob,
    save_model_every_n_iter, path_to_model_gan, path_to_model_ext_class)


