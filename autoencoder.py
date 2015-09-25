#!/usr/bin/env python
# encoding: utf-8


import sys
import numpy as np
import theano.tensor as T
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
from optparse import OptionParser
from theano import shared, function
from matplotlib.pyplot import subplot
from sklearn.preprocessing import scale
from sklearn.datasets import fetch_mldata
from sklearn.feature_extraction import image
from sklearn.utils import shuffle
from utils import *


def kl_divergence(rho, rho_cap):
    """TODO: Docstring for kl_divergence.

    :rho: TODO
    :rho_cap: TODO
    :returns: TODO

    """
    kl = T.sum(rho * T.log(rho/rho_cap)
            + (1. - rho) * T.log((1. - rho)/(1. - rho_cap)))

    return kl


def ae_encode(x, theta1, b1):
    """TODO: Docstring for ae_encode.
    :returns: TODO

    """
    neuron = T.nnet.sigmoid(T.dot(x, theta1) + b1)
    return neuron


def ae_decode(neuron, theta2, b2):
    """TODO: Docstring for ae_decode.
    :returns: TODO

    """
    outputs = T.nnet.sigmoid(T.dot(neuron, theta2) + b2)
    return outputs


def train_autoencode(X, hidrt, iter_num, alpha, decay, beta, rho):
    """TODO: Docstring for autoencode.

    :returns: TODO

    """
    input_n = X.shape[1]
    hidden_n = hidrt**2
    m = X.shape[0]
    params1 = initial_params(input_n, hidden_n)
    params2 = initial_params(hidden_n, input_n)

    x = T.matrix('x')
    theta1= shared(params1[0], name='theta1', borrow=True)
    b1 = shared(params1[1], name='b1', borrow=True)
    theta2= shared(params2[0], name='theta2', borrow=True)
    b2 = shared(params2[1], name='b2', borrow=True)
    neuron = ae_encode(x, theta1, b1)
    prediction = ae_decode(neuron, theta2, b2)
    rho_cap = T.sum(neuron, 0)/m
    # xent = (1./(2. * m)) * T.sum(T.sqr(prediction - x))
    xent = least_square(x, prediction, m)
    # xent = 0.5 * T.mean(T.sqr(prediction - x))
    cost = xent + (decay/(2.)) * (T.sum(T.sqr(theta1))+T.sum(T.sqr(theta2)))
    cost = cost + beta * kl_divergence(rho, rho_cap)
    grad = T.grad(cost, [theta1, b1, theta2, b2])

    trainit = init_gd_trainer(inputs=[x,], outputs=[prediction, xent, T.mean(neuron)],
                              name='trainit', params=[theta1, b1, theta2, b2],
                              grad=grad, alpha=alpha)
    # predict = function(inputs=[x], outputs=prediction, name = "predict")

    if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
            trainit.maker.fgraph.toposort()]):
        print('Used the cpu')
    elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
              trainit.maker.fgraph.toposort()]):
        print('Used the gpu')
    else:
        print('ERROR, not able to tell if theano used the cpu or the gpu')
        print(trainit.maker.fgraph.toposort())

    # for i in range(iter_num):
    #     pred, err = trainit(X)
    #     print 'iter: %f, err: %f\n' % (i, err)

    # for i in range(m):
    #     pred, err = trainit(X[i,:].reshape(1, visrt**2))
    #     print 'iter: %f, err: %f\n' % (i, err)

    # for i in range(iter_num*20):
    #     idx = int(np.random.random(1)*(m-1000))
    #     pred, err = trainit(X[idx:idx+1000,:])
    #     print 'iter: %f, err: %f\n' % (i, err)

    for i in range(iter_num):
        pred, err, mon = trainit(X)
        if i%100 == 0:
            print 'iter: %f, err: %f, mean of neuron: %f\n' % (i, err, mon)

    return theta1, b1, theta2, b2


def visualize_ae_image(theta1, filename, visrt, hidrt, processor, *arg):
    """TODO: Docstring for visualize_ae.

    :theta1: TODO
    :theta2: TODO
    :returns: TODO

    """
    max_act = (theta1/T.sqrt(T.sum(T.sqr(theta1), axis=0))).T
    cal_x = function(inputs=[], outputs=max_act, name='cal_x')

    x = cal_x().reshape(hidrt**2, visrt, visrt)
    x = processor(x, *arg)
    # print np.dot(cal_x(), theta1.get_value())
    # print np.sum(x[0, :, :]**2)
    # for i in range(1, hidrt**2+1):
    #     ax = subplot(hidrt, hidrt, i)
    #     # ax.imshow(x[i-1, :, :], cmap=plt.get_cmap('gray'))
    #     ax.imshow(x[i-1, :, :], cmap=plt.get_cmap('gray'), interpolation='nearest', aspect='equal')
    #     ax.axis('off')

    img = Image.fromarray(tile_raster_images(
        X=x,
        img_shape=(visrt, visrt), tile_shape=(hidrt, hidrt),
        tile_spacing=(1, 1)))
    img = img.resize((500,500))
    img.save(filename)

    return x


def get_patches():
    """TODO: Docstring for get_patches.
    :returns: TODO

    """
    a, b = {}, []
    sio.loadmat('starter/IMAGES.mat', a)
    for i in range(samples):
        seed = int(np.random.random(1)*10)
        ima = a['IMAGES'][:,:,seed]
        startx = int(np.random.random(1)*504)
        starty = int(np.random.random(1)*504)
        b.append(ima[startx:startx+visrt,starty:starty+visrt])

    return np.array(b, dtype='float32')


def process_grey(img):
    """TODO: Docstring for process_grey.
    :returns: TODO

    """
    img = img.astype('float32')
    img = img - np.mean(img)
    pstd = 3 * np.std(img)
    img[img>pstd] = pstd
    img[img<-pstd] = -pstd
    img = img/pstd

    return (img + 1) * 0.4 + 0.1


def extract_patches():
    """TODO: Docstring for extract_patches.
    :returns: TODO

    """
    a, b = {}, np.empty((0,visrt,visrt))
    sio.loadmat('starter/IMAGES.mat', a)
    for i in range(samples/1000):
        ima = a['IMAGES'][:,:,i%10]
        b = np.concatenate((b, image.extract_patches_2d(ima, (visrt,visrt), max_patches=1000, random_state=6)))

    b = np.array(b)

    return process_grey(b)


def test_edge_detector():
    """TODO: Docstring for test_edge_detector.
    :returns: TODO

    """
    iter_num = 1000
    alpha = 1.
    decay = 0.0033 # the performance of 0.0001 is not so good.
    beta = 3
    rho = 0.1
    visrt = 8
    hidrt = 5

    X = np.loadtxt(open("data.txt","rb"),delimiter=",",skiprows=0).T.astype('float32')
    theta1, b1, theta2, b2 = train_autoencode(X, hidrt, iter_num, alpha, decay, beta, rho)

    return visualize_ae_image(theta1, 'filters_corruption.png', visrt, hidrt, process_grey)


def test_mnist():
    """TODO: Docstring for test_mnist.
    :returns: TODO

    """
    iter_num = 400
    alpha = 1.
    decay = 0.0002
    beta = 3
    rho = 0.01
    visrt = 28
    hidrt = 14

    mnist = fetch_mldata('MNIST original', data_home='./')
    mnist = scale(mnist.data[mnist.target>=5,:].astype(float)).astype('float32')
    theta1, b1, theta2, b2 = train_autoencode(mnist, hidrt, iter_num, alpha, decay, beta, rho)

    return visualize_ae_image(theta1, 'filters_corruption.png', visrt, hidrt, process_grey)


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    test_mnist()
    # test_edge_detector()


if __name__ == '__main__':
    main()
