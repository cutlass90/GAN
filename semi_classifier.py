import time
from tqdm import tqdm

import tensorflow as tf
import numpy as np

from model_abstract import Model
from GAN import plot_samples

class SemiClassifier(Model):

    # --------------------------------------------------------------------------
    def __init__(self, gan, ext_classifier, do_train):

        self.gan = gan
        self.ext_classifier = ext_classifier
        self.do_train = do_train


    # --------------------------------------------------------------------------
    def train_model(self, image_provider, labeled_data_loader, test_data_loader, batch_size,
        weight_decay, n_iter, learn_rate_start, learn_rate_end, keep_prob,
        save_model_every_n_iter, path_to_model_gan, path_to_model_ext_class):

        start_time = time.time()
        for current_iter in tqdm(range(n_iter)):
            learn_rate = self.scaled_exp_decay(learn_rate_start, learn_rate_end,
                n_iter, current_iter)

            images = image_provider.next_batch(batch_size)
            train_batch = labeled_data_loader.next_batch(batch_size)

            # train GAN
            self.gan.train_disc_step(images, keep_prob, weight_decay, learn_rate)
            self.gan.train_gen_step(keep_prob, weight_decay, batch_size, learn_rate)
            self.gan.train_class_step(train_batch[0], train_batch[1], keep_prob,
                weight_decay, learn_rate)

            if current_iter%200 == 0:
                self.gan.save_disc_summaries(images, keep_prob, weight_decay, True,
                    self.gan.train_writer, current_iter)
                self.gan.save_gen_summaries(keep_prob, weight_decay, batch_size, True,
                    self.gan.train_writer, current_iter)
                self.gan.save_class_summaries(train_batch[0], train_batch[1], keep_prob,
                    weight_decay, True, self.gan.train_writer, current_iter)

                test_batch = test_data_loader.next_batch(batch_size)
                self.gan.save_disc_summaries(test_batch[0], keep_prob, weight_decay, False,
                    self.gan.test_writer, current_iter)
                self.gan.save_gen_summaries(keep_prob, weight_decay, batch_size, False,
                    self.gan.test_writer, current_iter)
                self.gan.save_class_summaries(test_batch[0], test_batch[1], keep_prob, weight_decay,
                    False, self.gan.test_writer, current_iter)
            if (current_iter+1)%5000 == 0:
                samples = self.gan.sample()
                plot_samples(samples, current_iter)

            # train ext_classifier
            gan_images, gan_labels = self.gan.sample_data(batch_size)
            self.ext_classifier.train_step(gan_images, gan_labels, weight_decay, learn_rate,
                keep_prob)
            if current_iter%200 == 0:
                self.ext_classifier.save_summaries(gan_images, gan_labels, weight_decay,
                    keep_prob, True, self.ext_classifier.train_writer, current_iter)
                self.ext_classifier.save_summaries(test_batch[0], test_batch[1],
                    weight_decay, 1, False, self.ext_classifier.test_writer, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.gan.save_model(path=path_to_model_gan, sess=self.gan.sess,
                    step=current_iter+1)
                self.ext_classifier.save_model(path=path_to_model_ext_class,
                    sess=self.ext_classifier.sess, step=current_iter+1)

        self.gan.save_model(path=path_to_model_gan, sess=self.gan.sess,
            step=current_iter+1)
        self.ext_classifier.save_model(path=path_to_model_ext_class,
            sess=self.ext_classifier.sess, step=current_iter+1)

        print('\nTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))
