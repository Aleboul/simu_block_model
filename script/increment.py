from cmath import log
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

def crit(X, w, cols):
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

    value = log_rank(X, w)

    _value_ = []
    for c in clust:
        index = np.where(cols == c)[0]
        _X = X[:,index]
        w_c = w[index]
        wei_clu = np.sum(w_c) / np.sum(w) 
        _value = log_rank(_X, w_c / np.sum(w_c))
        _value_.append(wei_clu * _value)

    return -(value - np.sum(_value_))

def split(X):

    S = np.arange(X.shape[1])
    colsind = S
    colsind[0] = -1
    d = X.shape[1]
    cols = np.ones(d)
    cols[0] = -1
    w = np.repeat(1/d, d)

    conv_pick = crit(X,w,cols)
    convind   = crit(X,w,colsind)
    print(conv_pick)

    C = np.where(cols == -1)[0]
    R = np.where(cols == 1)[0]

    for i in R :
        colsind[i] = -1
        _convind = crit(X,w,colsind)
        print(convind - _convind)
        if (convind - _convind >  + 0.01):
            convind = _convind
            cols[i] *= -1
            R = np.delete(R, np.where(R == i)[0])
            conv_pick = crit(X,w,cols)
            #break
        else:
            colsind[i] = i
    print(conv_pick)
    return R

d1 = 20
d2 = 20

cols_1 = np.repeat(-1,d1)
cols_2 = np.repeat(1,d2)
cols = np.hstack([cols_1,cols_2])

copula_1 = Logistic(theta = 0.4, n_sample = 200, d = d1)
copula_2 = Logistic(theta = 0.8, n_sample = 200, d = d2)

sample_1 = copula_1.sample_unimargin()
sample_2 = copula_2.sample_unimargin()

sample = np.hstack((sample_1, sample_2))
w = np.repeat(1/(d1+d2), d1+d2)
print(crit(sample,w, cols))

col = list(np.arange(d1+d2))
rand = random.sample(col, d1+d2)
sample = sample[:,rand]
cols = split(sample)

print(cols)

print(sample[cols])