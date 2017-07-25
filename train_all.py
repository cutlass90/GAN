
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from GAN import GAN
from classifier import Classifier
from semi_classifier import SemiClassifier

class ImageProvider:
    def __init__(self, mnist):
        self.mnist = mnist
    def next_batch(self, batch_size):
        return mnist.train.next_batch(batch_size)[0]


labeled_size = 60000-128*10
batch_size = 128
weight_decay = 2e-2
n_iter = 500000
learn_rate_start = 1e-2
learn_rate_end = 1e-4
keep_prob = 1
save_model_every_n_iter = 35000
path_to_model_gan = 'modes/gan'
path_to_model_ext_class = 'models/class'



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True,
    validation_size=labeled_size)

ext_classifier = Classifier(input_dim=784, n_classes=10, do_train=True,
    scope='ext_classifier')

gan = GAN(do_train=True, input_dim=784, n_classes=10, z_dim=20, scope='GAN')
semi_classifier = SemiClassifier(gan, ext_classifier, do_train=True)

image_provider = ImageProvider(mnist)
labeled_data_loader = mnist.validation
test_data_loader = mnist.test

semi_classifier.train_model(image_provider, labeled_data_loader, test_data_loader,
    batch_size, weight_decay, n_iter, learn_rate_start, learn_rate_end, keep_prob,
    save_model_every_n_iter, path_to_model_gan, path_to_model_ext_class)


