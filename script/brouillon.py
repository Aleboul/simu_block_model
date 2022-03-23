import numpy as np
import matplotlib.pyplot as plt
import random
plt.style.use('seaborn-whitegrid')

import sys
sys.path.insert(0, '/home/aboulin/')

from CoPY.src.rng.evd import Logistic, Asymmetric_logistic
from CoPY.src.rng.utils import simplex
from sklearn.cluster import KMeans

np.random.seed(42)

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

def critind(X, w, cols):
    """ evalutation of the criteria

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

    ### Evaluate the cluster as a whole

    _value_ = []
    for c in clust:
        index = np.where(cols == c)[0]
        _X = X[:,index]
        w_c = w[index]
        wei_clu = np.sum(w_c) / np.sum(w) 
        _value = log_rank(_X, w_c / np.sum(w_c))
        _value_.append(wei_clu * _value)

    print(_value_)

    return np.sum(_value_)

d1 = 20
d2 = 20

cols_1 = np.repeat(-1,d1)
cols_2 = np.repeat(1,d2)
cols = np.hstack([cols_1,cols_2])

j = 30

colind = np.arange(d1+d2)
colind[1] = 0
colind[j] = 0

copula_1 = Logistic(theta = 0.6, n_sample = 200, d = d1)
copula_2 = Logistic(theta = 0.8, n_sample = 200, d = d2)

sample_1 = copula_1.sample_unimargin()
sample_2 = copula_2.sample_unimargin()

sample = np.hstack((sample_1, sample_2))

w = np.repeat(1/(d1+d2), d1+d2)

print(log_rank(sample[:,[0,j]], [1/2,1/2]))

print(wmado(sample[:,[0,j]], [1/2,1/2]))

#print(critind(sample, w, colind))