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

np.random.seed(41)

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
    """Split a set S into two independent subsets

    Input
    -----
        S (list[]) : The set \{1,\dots, d\}
        X (matrix)

    Output
    ------
        S_1, S_2
    """
    print('...computing dissi matrix...')
    dissi_mat = diss(X)
    print('...doing Kmeans...')
    kmeans = KMeans(n_clusters=2, random_state=0).fit(dissi_mat)
    cols = 2*kmeans.labels_-1
    print(cols)
    w = np.repeat(1/X.shape[1], X.shape[1])
    conv_pick = crit(X,w,cols)
    count = 0
    print('...checking independency...')
    while(conv_pick > 0.0 and count < X.shape[1]):
        S = list(np.arange(X.shape[1]))
        count = 0
        for i in S:

            cols[i] *= -1
            _conv_pick = crit(X,w,cols)
            if (conv_pick > _conv_pick):
                conv_pick = _conv_pick
                break
            else:
                cols[i] *= -1
            count += 1
            print(count)
        

    return cols

#def split(X):
#
#    S = np.arange(X.shape[1])
#    colsind = S
#    colsind[0] = -1
#    d = X.shape[1]
#    cols = np.ones(d)
#    cols[0] = -1
#    w = np.repeat(1/d, d)
#
#    conv_pick = crit(X,w,cols)
#    convind   = crit(X,w,colsind)
#
#    C = np.where(cols == -1)[0]
#    R = np.where(cols == 1)[0]
#
#    while(conv_pick > 0.012):
#        for i in R :
#            colsind[i] = -1
#            _convind = crit(X,w,colsind)
#            if (convind > _convind + 0.01):
#                convind = _convind
#                cols[i] *= -1
#                R = np.delete(R, np.where(R == i)[0])
#                conv_pick = crit(X,w,cols)
#                break
#            else:
#                colsind[i] = i
#        print(conv_pick)
#    return Rasy

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
    Test
"""


d1 = 10
d2 = 10
d3 = 10

cols_1 = np.repeat(-1,d1)
cols_2 = np.repeat(1,d2)
cols_3 = np.repeat(1,d3)
cols = np.hstack([cols_1,cols_2,cols_3])

copula_1 = Logistic(theta = 0.8, n_sample = 500, d = d1)
copula_2 = Logistic(theta = 0.8, n_sample = 500, d = d2)
copula_3 = Logistic(theta = 0.8, n_sample = 500, d = d3)

sample_1 = copula_1.sample_unimargin()
sample_2 = copula_2.sample_unimargin()
sample_3 = copula_3.sample_unimargin()
w = np.repeat(1/(d1+d2+d3), d1+d2+d3)
sample = np.hstack((sample_1, sample_2, sample_3))
print(crit(sample,w,cols))

label = split(sample)
print(label)

grp_1 = np.where(label == -1)[0]

label = split(sample[:,grp_1])

print(label)

"""
    test asy log
"""

#asy = [0, 0, 0, 0, [0,0], [0,0], [0,0], [0,0], [0,0], [0,0],[.2,.1,.2], [.1,.1,.2], [.3,.4,.1], [.2,.2,.2], [.4,.6,.2,.5]]
#
#copula = Asymmetric_logistic(theta =[0.1,0.1,0.5,0.9], asy = asy, n_sample = 200, d=4)
#sample = copula.sample_unimargin()
#dissmat = diss(sample)
#print(dissmat)
#
#col = split(sample)
#
#print(col)