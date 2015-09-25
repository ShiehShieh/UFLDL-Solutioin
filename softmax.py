#!/usr/bin/env python
# -*- coding: utf-8 -*-


import theano.tensor as T
from theano import shared
from utils import *
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cross_validation import train_test_split


def softmax_predict(x, weight, b):
    """TODO: Docstring for softmax_predict.

    :arg1: TODO
    :returns: TODO

    """
    z = T.dot(x, weight) + b
    pred = T.nnet.softmax(z)

    return pred


def softmax2class_threshold(pred, threshold):
    """TODO: Docstring for softmax2class.
    :returns: TODO

    """
    pred[pred>=threshold] = 1
    pred[pred<threshold] = 0

    return pred


def softmax2class_max(pred):
    """TODO: Docstring for softmax2class_max.
    :returns: TODO

    """
    return np.argmax(pred, axis=1)


def train_softmax(X, y, iter_num, alpha, decay):
    """TODO: Docstring for train_softmax.
    :returns: TODO

    """
    input_n, output_n = X.shape[1], y.shape[1]
    m = X.shape[0]
    params = initial_params(input_n, output_n)

    t_X, t_y = T.matrix(), T.matrix()
    theta = shared(params[0], name='theta', borrow=True)
    b = shared(params[1], name='b', borrow=True)

    z = softmax_predict(t_X, theta, b)
    J = -T.sum(T.log2(T.exp(T.sum(z * t_y, 1)) / T.sum(T.exp(z), 1)))/m \
        + (decay / (2.0 * m)) * T.sum(theta ** 2.0)
    grad = T.grad(J, [theta, b])

    trainit = init_gd_trainer(inputs=[t_X, t_y], outputs=[z, J,], name='trainit',
                              params=[theta, b,], grad=grad, alpha=alpha)

    for i in range(iter_num):
        pred, err = trainit(X, y)
        if i%100 == 0:
            print 'iter: %f, err: %f\n' % (i, err)

    return theta, b


def main():
    """TODO: Docstring for main.

    :arg1: TODO
    :returns: TODO

    """
    alpha = 0.6
    iter_num = 1200
    decay = 0.02 # the performance of 0.0001 is not so good.
    enc = OneHotEncoder(sparse=False)
    mnist = fetch_mldata('MNIST original', data_home='./')
    x_train, x_test, y_train, y_test = \
            train_test_split(scale(mnist.data.astype(float)).astype('float32'),
                             mnist.target.astype('float32'),
                             test_size=0.15, random_state=0)
    y_train = enc.fit_transform(y_train.reshape(y_train.shape[0],1)).astype('float32')
    theta, b = train_softmax(x_train, y_train, iter_num, alpha, decay)

    x = T.matrix()
    f = function([x], [softmax_predict(x, theta, b)])

    pred = softmax2class_max(f(x_test)[0])
    print accuracy_score(y_test, pred)
    print classification_report(y_test, pred)
    print confusion_matrix(y_test, pred)


if __name__ == "__main__":
    main()
