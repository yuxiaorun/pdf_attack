# -*- coding: UTF-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
import data_process
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans_model import make_basic_DNN
from cleverhans.utils import AccuracyReport, set_log_level

import os

FLAGS = flags.FLAGS


def mnist_tutorial(nb_epochs=1000, batch_size=128,
                   learning_rate=0.0001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   num_threads=None, load_model=True):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Get MNIST test data

    X_train, X_test= np.load('mdata_10000.npy')
    Y_train, Y_test= [[1,0] for x in range(X_train.shape[0])]
    # Use label smoothing

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 5000))
    y = tf.placeholder(tf.float32, shape=(None, 2))

    model_path = "models/DNN"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'filename': "DNN_clean_model.ckpt",
        'train_dir': model_path
    }
    ckpt = tf.train.get_checkpoint_state(train_params['train_dir'])
    ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path
    fgsm_params = {'eps': 0.1,
                   'clip_min': 0.,
                   'clip_max': 1.}
    rng = np.random.RandomState([2017, 8, 30])

    if clean_train:
        model = make_basic_DNN()
        preds = model.get_probs(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            print('Test accuracy on legitimate examples: %0.4f' % acc)
        if load_model is False:
            print('start train')
            model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                        args=train_params, rng=rng,save=True)
        else:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_path)
            print("Model loaded from: {}".format(ckpt_path))
            evaluate()

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        fgsm = FastGradientMethod(model, sess=sess)
        adv_x, normalized_grad = fgsm.generate(x, **fgsm_params)         #此处在fgsm.generate函数中有改动。返回多了一个梯度值
        influence=tf.reduce_mean(tf.abs(normalized_grad), 0)              #得到梯度绝对值
        preds_adv = model.get_probs(adv_x)

        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)

        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc
        np.save('grads.npy', influence.eval({x: X_test},session=sess))      #保存梯度绝对值。
        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc


    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 1000, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

tf.app.run()