#!/usr/bin/env python
# -*- coding: utf-8 -*-


import re
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from autoencoder import *


def get_patchs(corpus, patches, length):
    """TODO: Docstring for get_patchs.
    :returns: TODO

    """
    a = []
    punct = re.compile(r'([^A-Za-z0-9 ])')
    for i in range(patches):
        seed = int(np.random.random(1)*len(corpus))
        article = punct.sub('', corpus[seed])
        start = int(np.random.random(1)*(len(article)-length-1))
        a.append([ord(s) for s in article[start:start+length]])

    return np.array(a, dtype='float32')


def normalize_char(corpus, max_char, min_char):
    """TODO: Docstring for normalize_char.
    :returns: TODO

    """
    return (corpus-min_char)/(max_char-min_char)


def denormalize_char(corpus, max_char, min_char):
    """TODO: Docstring for denormalize_char.
    :returns: TODO

    """
    return (np.abs(corpus / np.max(corpus) * (max_char - min_char)) + min_char).astype(int)


def visualize_ae_char(theta1, processor, *arg):
    """TODO: Docstring for visualize_ae_char.

    :arg1: TODO
    :returns: TODO

    """
    max_act = (theta1/T.sqrt(T.sum(T.sqr(theta1), axis=0))).T
    cal_x = function(inputs=[], outputs=max_act, name='cal_x')
    x = cal_x()
    x = processor(x, *arg)
    print x
    with open('visualize_ae_char.txt', 'w') as tmp:
        for row in x:
            tmp.write(''.join([chr(a) for a in row])+'\n')


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    iter_num = 2200
    alpha = 1.
    weight = 0.0001 # the performance of 0.0001 is not so good.
    beta = 3
    rho = 0.01

    dataset = fetch_20newsgroups(data_home='./dataset').data
    X = get_patchs(dataset, 10000, 64)
    max_char = np.max(X)
    min_char = np.min(X)
    X = normalize_char(X, max_char, min_char)
    theta1, b1, theta2, b2 = auto_encode(X, iter_num, alpha, weight, beta, rho)

    return visualize_ae_char(theta1, denormalize_char, max_char, min_char)


if __name__ == "__main__":
    main()
