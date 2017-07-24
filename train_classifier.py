
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from GAN import GAN
from classifier import Classifier

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

ext_classifier = Classifier(input_dim=784, n_classes=10, do_train=True,
    scope='ext_classifier')
gan = GAN(do_train=False, input_dim=784, n_classes=10, z_dim=20, scope='GAN')
gan.load_model(path='models/', sess=gan.sess)

for it in range(50000):
    images, labels = gan.sample_data(128)
    ext_classifier.train_step(images, labels, 0.02, 0.0005)
    images, labels = mnist.test.next_batch(128)
    ext_classifier.save_summaries(images, labels, 0.02, False, ext_classifier.test_writer, it)

