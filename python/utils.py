"""This file is a lib package with all the dependants"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import metrics
# from torch.utils.data.dataloader import DataLoader


def do(tau, X):
    """shrinkage operator for singular values"""
    [U, S, V] = torch.svd(X)
    return U*torch.diag(so(tau, S))*V.t()


def so(tau, X):
    # shrinkage  operator
    return np.sign(X) * np.maximum(abs(X) - tau, 0)


def load_data(name='EYB'):
    if name == 'AR': return load_ar()
    if name == 'EYB': return load_eyb()
    if name == 'coil': return load_coil()


def load_ar():
    data = pd.read_csv('AR55x40.csv', header=None).to_numpy()
    labels = pd.read_csv('AR55x40_labels.csv', header=None).to_numpy()
    return data, labels


def load_eyb():
    data = pd.read_csv('EYB32x32.csv',header=None).to_numpy()
    labels = pd.read_csv('EYB32x32_labels.csv',header=None).to_numpy()
    return data, labels


def load_coil():
    data = pd.read_csv('coil20.csv',header=None).to_numpy()
    labels = pd.read_csv('coil20_labels.csv',header=None).to_numpy()
    return data, labels


def data_gen(data, labels, optdata):
    """This function will normalize all the data,
        adding outlier, only l1 type, l21not implemented
        splitting data to train and test """

    M, N = data.shape  # M is dimension, N is sample size
    MN = M * N
    oMN = int(optdata['o_per'] * MN)  # how mnay are corrupted
    np.random.seed(optdata['seed']+100)
    ind_E = np.arange(MN)  # serialize the index
    np.random.shuffle(ind_E)  # shuffle index
    data[ind_E[:oMN]] = np.random.randint(0, 255, oMN)  # randomly to corrupt

    data_norm = data / np.sqrt((data * data).sum(0))  # normalize per column

    # split data according to labels
    C = labels.max()
    np.random.seed(optdata['seed'])
    indx = np.arange(C)
    np.random.shuffle(indx)
    if optdata['dataset'] == 'EYB':
        tr_len = 30
    if optdata['dataset'] == 'coil':
        tr_len = 40
    if optdata['dataset'] == 'AR':
        tr_len = 16

    xtr = np.array([])
    xtr_labels = np.array([])
    xte = np.array([])
    xte_labels = np.array([])

    for i in range(C):
        pass





