#!/usr/bin/env python
# -*- coding: utf-8 -*-


import theano.tensor as T
from theano import function
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from softmax import train_softmax, softmax_predict, softmax2class_max, cost4softmax
from autoencoder import train_autoencode, ae_encode, visualize_ae_image, process_grey
from utils import *


def sae_predict(X, weights):
    """TODO: Docstring for sae_predict.
    :returns: TODO

    """
    inp = T.matrix(name='inp')
    for idx, hp in enumerate(weights):
        if idx == 0:
            res = ae_encode(inp, hp[0], hp[1])
        elif idx != len(weights)-1:
            res = ae_encode(res, hp[0], hp[1])
        else:
            res = softmax_predict(res, hp[0], hp[1])

    f = function(inputs=[inp,], outputs=[res,], name='f')

    return f(X)[0]


def pretrain_sae(X, hyper_params):
    """TODO: Docstring for pretrain_sae.

    :arg1: TODO
    :returns: TODO

    """
    params = []
    x = T.matrix()

    layer_input = X
    for idx, hidsize in enumerate(hyper_params['hidden_layers_sizes']):
        W, b = train_autoencode(layer_input, int(np.sqrt(hidsize)),
                                hyper_params['iter_nums'][idx],
                                hyper_params['alphas'][idx],
                                hyper_params['decays'][idx],
                                hyper_params['betas'][idx],
                                hyper_params['rhos'][idx])[:2]
        params.append((W, b))
        visualize_ae_image(W, 'sae_%d.png' % (idx), int(np.sqrt(W.get_value().shape[0])),
                           int(np.sqrt(W.get_value().shape[1])), process_grey)
        f = function(inputs=[x,], outputs=[ae_encode(x, W, b)], name='f')
        layer_input = f(layer_input)[0]

    return params, layer_input


def sae_extract(x, weights):
    """TODO: Docstring for sae_extract.
    :returns: TODO

    """
    pred = x
    for hp in weights:
        pred = ae_encode(pred, hp[0], hp[1])

    return pred


def finetune_sae(X, y, weights, finetune_iter, alpha, decay):
    """TODO: Docstring for finetune_sae.

    :arg1: TODO
    :returns: TODO

    """
    m = X.shape[0]
    t_x = T.matrix(name='x')
    t_y = T.matrix(name='x')
    pred = sae_extract(t_x, weights[:-1])
    pred = softmax_predict(pred, *weights[-1]) # weights[-1][0], weights[-1][1])
    cost = cost4softmax(pred, t_y, m, decay, weights[-1][0])

    unroll = []
    for hp in weights:
        unroll.append(hp[0])
        unroll.append(hp[1])
    grad = T.grad(cost, unroll)

    trainit = init_gd_trainer(inputs=[t_x, t_y], outputs=[pred, cost],
                              name='trainit', params=unroll,
                              grad=grad, alpha=alpha)

    for i in range(finetune_iter):
        pred, err = trainit(X, y)
        if i%100 == 0:
            print 'iter: %f, err: %f\n' % (i, err)

    return [(unroll[2*idx], unroll[2*idx+1]) for idx in range(len(unroll)/2)]


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    alpha = 1.
    decay = 0.0006
    iter_num = 600
    finetune_iter = 220
    hyper_params = {
            'hidden_layers_sizes':[196,], 'iter_nums':[400,],
            'alphas':[1.,], 'decays':[0.003,],
            'betas':[3,], 'rhos':[0.1,]
            }

    enc = OneHotEncoder(sparse=False)
    mnist = fetch_mldata('MNIST original', data_home='./')
    x_train, x_test, y_train, y_test = \
            train_test_split(scale(mnist.data.astype(float)).astype('float32'),
                             mnist.target.astype('float32'),
                             test_size=0.5, random_state=0)
    x_unlabeled = scale(mnist.data[mnist.target>=5,:].astype(float)).astype('float32')
    y_train = enc.fit_transform(y_train.reshape(y_train.shape[0],1)).astype('float32')

    t_x = T.matrix()
    params, extracted = pretrain_sae(x_unlabeled, hyper_params)
    extracted = function(inputs=[t_x], outputs=[sae_extract(t_x, params)])(x_train)[0]
    params.append(train_softmax(extracted, y_train, iter_num, alpha, decay))
    weights = finetune_sae(x_train, y_train, params, finetune_iter, alpha, decay)

    all_label = np.array(range(0, 10))
    pred = all_label[softmax2class_max(sae_predict(x_test, weights))]
    print accuracy_score(y_test, pred)
    print classification_report(y_test, pred)
    print confusion_matrix(y_test, pred)


if __name__ == "__main__":
    main()
