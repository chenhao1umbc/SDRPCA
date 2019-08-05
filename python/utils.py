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
    data = pd.read_csv('csvdata/AR55x40.csv', header=None).to_numpy()
    labels = pd.read_csv('csvdata/AR55x40_labels.csv', header=None).to_numpy().squeeze()
    return data, labels


def load_eyb():
    data = pd.read_csv('csvdata/EYB32x32.csv',header=None).to_numpy()
    labels = pd.read_csv('csvdata/EYB32x32_labels.csv',header=None).to_numpy().squeeze()
    return data, labels


def load_coil():
    data = pd.read_csv('csvdata/coil20.csv',header=None).to_numpy()
    labels = pd.read_csv('csvdata/coil20_labels.csv',header=None).to_numpy().squeeze()
    return data, labels


def data_gen(data, labels, optdata):
    """This function will normalize all the data,
        adding outlier, only l1 type, l21not implemented
        splitting data to train and test """

    M, N = data.shape  # M is dimension, N is sample size
    MN = M * N
    oMN = int(optdata['o_per'] * MN)  # how mnay are corrupted
    np.random.seed(optdata['rng']+100)
    ind_E = np.arange(MN)  # serialize the index
    np.random.shuffle(ind_E)  # shuffle index
    data.ravel()[ind_E[:oMN]] = np.random.randint(0, 255, oMN)  # randomly corrupt

    data_norm = data / np.sqrt((data * data).sum(0))  # normalize per column

    # split data according to labels
    C = labels.max()
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
        current_label = np.where(labels == i+1) # current_label is a tuple
        start_point = current_label[0][0]
        end_point = current_label[0][-1]

        data_norm = data_norm.T
        np.random.seed(optdata['rng'])
        np.random.shuffle(data_norm[start_point:end_point, :])  # only shuffle the first dimension
        data_norm = data_norm.T

        xtr = np.c_[xtr, data_norm[:, start_point:start_point+tr_len]] if xtr.size else data_norm[:, start_point:start_point+tr_len]
        xtr_labels = np.r_[xtr_labels, labels[start_point:start_point+tr_len]] if xtr_labels.size else labels[start_point:start_point+tr_len]
        xte = np.c_[xte, data_norm[:, start_point+tr_len:end_point]] if xte.size else data_norm[:, start_point+tr_len:end_point]
        xte_labels = np.r_[xte_labels, labels[start_point+tr_len:end_point]] if xte_labels.size else labels[start_point+tr_len:end_point]

    # return xtr, xtr_labels, xte, xte_labels
    return n2t(xtr), n2t(xtr_labels), n2t(xte), n2t(xte_labels)


def n2t(x):
    return torch.from_numpy(x).float()


def train_lrr(X, A, lamb, optdata):
    # initialization
    X = X.cuda() if optdata['use_gpu'] else X
    A = A.cuda() if optdata['use_gpu'] else A

    m, n = X.shape
    mu = 0.1 * lamb
    atx = A.t()@X
    inv_a = torch.inverse(A.t()@A + torch.eye(n)) if not optdata['use_gpu'] else torch.inverse(A.t()@A + torch.eye(n).cuda())

    J = torch.zeros(n, n).cuda() if optdata['use_gpu'] else torch.zeros(n, n)
    Z = torch.zeros(n, n).cuda() if optdata['use_gpu'] else torch.zeros(n, n)
    E = torch.zeros(m, n).cuda() if optdata['use_gpu'] else torch.zeros(m, n)
    Y1 = torch.zeros(m, n).cuda() if optdata['use_gpu'] else torch.zeros(m, n)
    Y2 = torch.zeros(n, n).cuda() if optdata['use_gpu'] else torch.zeros(n, n)
    diff = torch.zeros(optdata['max_iter'])

    # body of algorithm
    for i in range(optdata['max_iter']):
        # update J
        temp = Z + Y2 / mu;
        [U, sigma, V] = torch.svd(temp)
        svp = sigma[sigma>1/mu].shape[0]
        if svp > 1:
            sigma = sigma[:svp] - 1/mu
            J = U[:, :svp] @ sigma.diag() @ V[:, :svp].t()
        else:
            svp, sigma = 1, 0
            J = U[:, :svp] *sigma @ V[:, :svp].t()

        # update Z
        Z = inv_a @ (atx - A.t()@E+J+(A.t()@Y1-Y2) / mu)

        # update E
        xmaz = X - A @ Z
        temp = xmaz + Y1 / mu
        E_temp = np.maximum(0, (temp - lamb / mu).cpu().numpy())+np.minimum(0, (temp + lamb / mu).cpu().numpy())
        E = torch.from_numpy(E_temp) if not optdata['use_gpu'] else torch.from_numpy(E_temp).cuda()

        # when to stop
        leq1 = xmaz - E
        leq2 = Z - J
        stopC = np.maximum(leq1.abs().max().cpu().numpy(), leq2.abs().max().cpu().numpy())
        diff[i] = torch.from_numpy(np.array([stopC]))
        if i > 10 and (abs(diff[i] - diff[i - 10]) / abs(diff[i]) < 1e-3):
            break
        if stopC < optdata['tol']:
            break
        else:
            Y1 = Y1 + mu * leq1
            Y2 = Y2 + mu * leq2
            mu = np.minimum(optdata['max_mu'], mu * optdata['rho'])
    return Z, E



def get_prj(x):
    x0 = x- x.mean(1).reshape(x.shape[0], -1)
    u, s, _ = torch.svd(x0@x0.t())
    threshold = 1e4
    min_s = s.max()/threshold
    P = u[:, :s[s>min_s].shape[0]].t()
    return P.cpu()



