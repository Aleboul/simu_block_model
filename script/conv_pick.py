from cmath import log
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
plt.style.use('seaborn-whitegrid')

import sys
sys.path.insert(0, '/home/boulin/')

from COPPY.coppy.rng.evd import Logistic, Asymmetric_logistic, Husler_Reiss
from COPPY.coppy.rng.utils import simplex
from sklearn.cluster import KMeans


def ecdf(X):
    """ Compute uniform ECDF.
    
    Inputs
    ------
        X (np.array[float]) : array of observations

    Output
    ------
        Empirical uniform margin
    """

    index = np.argsort(X)
    ecdf = np.zeros(len(index))

    for i in index :
        ecdf[i] = (1.0 / X.shape[0]) * np.sum(X <= X[i])

    return ecdf

def ksi(X, w):
    """ Compute 
        ::math (\ln(-U_{i,1}) / w_1, \dots, \ln(-U_{i,d}) / w_d)

    Input
    -----
        X (np.array(float)) : d array of observation
                  w (float) : element of the simplex in R^d
    
    """
    d = X.shape[0]

    log_rank = np.log(X)

    _values_ = []
    for j in range(0,d):
        if w[j] == 0.0:
            _values_.append(10e6)
        else : 
            value = -log_rank[j] / w[j]
            _values_.append(value)

    return np.min(_values_)

def wmado(X, w) :
    """
        This function computes the w-madogram

        Inputs
        ------
        X (array([float]) of n_sample \times d) : a matrix
                                              w : element of the simplex
                            miss (array([int])) : list of observed data
                           corr (True or False) : If true, return corrected version of w-madogram
        
        Outputs
        -------
        w-madogram
    """

    Nnb = X.shape[1]
    Tnb = X.shape[0]
    V = np.zeros([Tnb, Nnb])
    for j in range(0, Nnb):
        X_vec = np.array(X[:,j])
        Femp = ecdf(X_vec)
        V[:,j] = np.power(Femp, 1/w[j])
    value_1 = np.amax(V,1)
    value_2 = (1/Nnb) * np.sum(V, 1)
    mado = (1/Tnb) * np.sum(value_1 - value_2)

    c = (1/Nnb)*np.sum(np.divide(w, 1 + np.array(w)))
    value = (mado + c) / (1-mado-c)
    return value

def log_rank(X, w):
    """ Compute Gudendorf-Segers estimator of the multivariate Pickands dependence function

    Input
    -----
        X (np.array(float)) : n x d array of observation
                  w (float) : element of the simplex in R^d

    Output
    ------
    
    """

    n = X.shape[0]
    d = X.shape[1]

    mat_rank = np.zeros((n,d))


    ### compute the matrix of rank

    for j in range(0,d):
        _rank = ecdf(X[:,j])
        mat_rank[:,j] = _rank

    _ksi_ = []
    for i in range(0,n):
        _input_ = mat_rank[i,:]
        _ksi_.append(ksi(_input_, w))
    
    return np.power(np.mean(_ksi_), -1)

def SECO(X, w, cols):
    """ evaluation of the criteria

    Input
    -----
        X (np.array(float)) : n x d array of observation
                  w (float) : element of the simplex
           cols (list(int)) : partition of the columns

    Output
    ------
        Evaluate (A - A_\Sigma)(w)
    
    """

    clust = np.unique(cols)
    d = X.shape[0]

    ### Evaluate the cluster as a whole

    value = d*wmado(X, w)

    _value_ = []
    for c in clust:
        index = np.where(cols == c)[0]
        _X = X[:,index]
        w_c = w[index]
        wei_clu = np.sum(w_c) / np.sum(w) 
        _value = wmado(_X, w_c / np.sum(w_c))
        _value_.append( d * wei_clu * _value)

    return -(value - np.sum(_value_))


def split(X):

    S = np.arange(X.shape[1])
    colsind = S
    colsind[0] = -1
    d = X.shape[1]
    cols = np.ones(d)
    cols[0] = -1
    w = np.repeat(1/d, d)

    conv_pick = SECO(X,w,cols)
    convind   = SECO(X,w,colsind)

    C = np.where(cols == -1)[0]
    R = np.where(cols == 1)[0]

    while(conv_pick > 0.012):
        for i in R :
            colsind[i] = -1
            _convind = SECO(X,w,colsind)
            if (convind > _convind + 0.01):
                convind = _convind
                cols[i] *= -1
                R = np.delete(R, np.where(R == i)[0])
                conv_pick = SECO(X,w,cols)
                break
            else:
                colsind[i] = i
        print(conv_pick)
    return R

def diss(X):

    d = X.shape[1]
    matrix = np.zeros([d,d])

    for i in range(0,d):
        for j in range(i, d):
            if i == j :
                matrix[i,i] = 0
            else :
                value_ = log_rank(X[:,[i,j]],[1/2,1/2])
                matrix[i,j] = value_
                matrix[j,i] = value_

    return matrix

"""
    Model's constant
"""

def adapt_const(delta, card, d,n):
    r = 2 * np.sqrt(2/n * np.log(4 * card / delta))
    value_ = 2 * (card+1)/ d * r
    return value_

"""
    Function for HR model
"""

def Gamma2Sigma(Gamma, k=1, full = False):
    d = Gamma.shape[0]
    if full:
        mat = 1/2 * (np.matrix(np.repeat(Gamma[:,k], d)).reshape((d,d)) + np.matrix(np.repeat(Gamma[:,k], d)).reshape((d,d)).T - Gamma)
        return mat
    else :
        index = np.arange(d)
        mat = 1/2 * (np.matrix(np.repeat(Gamma[index != k, :][:,k], d-1)).reshape((d-1,d-1)) + np.matrix(np.repeat(Gamma[index != k,:][:,k],d-1)).reshape((d-1,d-1)).T - Gamma[index != k, :][:, index != k])
        return mat

def Sigma2Gamma(Sigma, k = 0, full = False):
    # complete S
    if not full:
        d = Sigma.shape[0] + 1
        S_full = np.insert(Sigma, 0, np.repeat(0,d-1), axis = 0)
        S_full = np.insert(S_full, 0, np.repeat(0,d), axis = 1)

        shuffle = np.arange(d)
        shuffle[shuffle <= k] = shuffle[shuffle <= k] -1
        shuffle[0] = k
        shuffle = np.sort(shuffle)

        S_full[shuffle, :][:,shuffle]
    else:
        S_full = Sigma

    One = np.ones((1,d))
    D = np.diag(S_full).reshape((1,d))
    Gamma = One.T @ D + D.T @ One - 2 * S_full

    return Gamma

def sym_matrix(d, entry = 0.5, alea = True, a = 0.0, b=1.0):
    output = np.ones((d,d))
    for i in range(0,d):
        for j in range(0,i):
            if i == j:
                output[i,i] = 0.0
            else :
                if alea:
                    entry = np.random.uniform(a, b, size = 1)
                    output[i,j], output[j,i] = entry, entry
                else :
                    output[i,j], output[j,i] = entry, entry

    return np.matrix(output)
        
"""
    Test
"""
np.random.seed(41)
d = 400
n = 200
delta = 0.05

"""
    HR
"""

d1 = 24
d2 = 74
n_sample = 200
n_iter = 200

def iter_pick(d1,d2,n_sample, n_iter):
    """ Monte Carlo Simulation from Hüsler-Reiss model
    to see the behaviour of the SECO criteria.

    Input
    -----
        d1       : length of the first cluster
        d2       : length of the second cluster
        n_sample : observations
        n_iter   : number of iterations
    """
    output_store = []
    d = d1 + d2 + 2
    for n in range(0, n_iter):
        print(n)
        values = []
        # Generate first sample
        Sigma = sym_matrix(d1, alea = True, a = 0.1, b=0.2) 
        Sigma = Sigma @ Sigma.T
        Gamma = Sigma2Gamma(Sigma)
        copula1 = Husler_Reiss(n_sample = n_sample, d = d1+1, Sigma = Gamma)
        sample1 = copula1.sample_unimargin()
        # Generate second sample
        Sigma = sym_matrix(d2, alea = True, a = 0.01, b=0.2) 
        Sigma = Sigma @ Sigma.T
        Gamma = Sigma2Gamma(Sigma)
        copula2 = Husler_Reiss(n_sample = n_sample, d = d2+1, Sigma = Gamma)
        sample2 = copula2.sample_unimargin()
        # merge sample
        sample = np.hstack((sample1, sample2))
        # initialization
        w = np.repeat(1/(d), d)
        grp = np.arange(d)
        criteria = SECO(sample, w, grp)
        values.append(criteria)
        grp[0] = -1
        for j in range(1,d):
            grp[j] = -1
            criteria = SECO(sample, w,grp)
            if j >= d1+1:
                grp[j] = j
            values.append(criteria)
        output_store.append(values)
    column = np.arange(d)+1
    df = pd.DataFrame(output_store)
    return df

def iter_pick_logistic(d1,d2,n_sample, n_iter):
    """ Monte Carlo Simulation from Hüsler-Reiss model
    to see the behaviour of the SECO criteria.

    Input
    -----
        d1       : length of the first cluster
        d2       : length of the second cluster
        n_sample : observations
        n_iter   : number of iterations
    """
    output_store = []
    d = d1 + d2 + 2
    for n in range(0, n_iter):
        print(n)
        values = []
        # Generate first sample
        copula1 = Logistic(n_sample = n_sample, d = d1+1, theta = np.random.uniform(0.3,0.6,1))
        sample1 = copula1.sample_unimargin()
        # Generate second sample
        copula2 = Logistic(n_sample = n_sample, d = d2+1, theta = np.random.uniform(0.3,0.6,1))
        sample2 = copula2.sample_unimargin()
        # merge sample
        sample = np.hstack((sample1, sample2))
        # initialization
        w = np.repeat(1/(d), d)
        grp = np.arange(d)
        criteria = SECO(sample, w, grp)
        values.append(criteria)
        grp[0] = -1
        for j in range(1,d):
            grp[j] = -1
            criteria = SECO(sample, w,grp)
            if j >= d1+1:
                grp[j] = j
            values.append(criteria)
        output_store.append(values)
    column = np.arange(d)+1
    df = pd.DataFrame(output_store)
    return df

df = iter_pick(d1,d2,n_sample,n_iter)
df.to_csv('SECO_25_75_HR.csv')
#print(df)
#print(df.iloc[0])
#fig, ax = plt.subplots()
#ax.plot(np.arange(d1+d2 + 2)+ 1, df.iloc[0])
#plt.savefig('pick_crit.pdf')