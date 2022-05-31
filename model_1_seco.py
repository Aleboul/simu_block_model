import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
plt.style.use('seaborn-whitegrid')

from coppy.rng.evd import Logistic
from sklearn.model_selection import train_test_split

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

def theta(R) :
    """
        This function computes the w-madogram

        Inputs
        ------
        R (array([float]) of n_sample \times d) : rank's matrix
                                              w : element of the simplex
                            miss (array([int])) : list of observed data
                           corr (True or False) : If true, return corrected version of w-madogram
        
        Outputs
        -------
        w-madogram
    """

    Nnb = R.shape[1]
    Tnb = R.shape[0]
    V = np.zeros([Tnb, Nnb])
    for j in range(0, Nnb):
        V[:,j] = R[:,j]
    value_1 = np.amax(V,1)
    value_2 = (1/Nnb) * np.sum(V, 1)
    mado = (1/Tnb) * np.sum(value_1 - value_2)

    value = (mado + 1/2) / (1/2-mado)
    return value

def SECO(R_1, R_2, clst):
    """ evaluation of the criteria

    Input
    -----
        R (np.array(float)) : n x d rank matrix
                  w (float) : element of the simplex
           cols (list(int)) : partition of the columns

    Output
    ------
        Evaluate (theta - theta_\Sigma) 
    
    """

    d = R_1.shape[0]

    ### Evaluate the cluster as a whole

    value = theta(R_1)

    _value_ = []
    for key, c in clst.items():
        _R_2 = R_2[:,c]
        _value_.append(theta(_R_2))

    print("theta", value)
    print("sum_theta", _value_)
    return np.abs(np.sum(_value_) - value)

def find_max(M, S):
    mask = np.zeros(M.shape, dtype = bool)
    values = np.ones((len(S),len(S)),dtype = bool)
    mask[np.ix_(S,S)] = values
    np.fill_diagonal(mask,0)
    max_value = M[mask].max()
    i , j = np.where(np.multiply(M, mask * 1) == max_value) # Sometimes doublon happens for excluded clusters, if n is low
    return i[0], j[0]


def clust(Theta, n, alpha = None):
    """ Performs clustering in AI-block model
    
    Inputs
    ------
        Theta : extremal correlation matrix
        alpha : threshold, of order sqrt(ln(d)/n)
    
    Outputs
    -------
        Partition of the set \{1,\dots, d\}
    """
    d = Theta.shape[1]

    # Initialisation

    S = np.arange(d)
    l = 0

    if alpha is None:
        alpha = 2 * np.sqrt(np.log(d)/n)
    
    cluster = {}
    while len(S) > 0:
        l = l + 1
        if len(S) == 1:
            cluster[l] = np.array(S)
        else :
            a_l, b_l = find_max(Theta, S)
            if Theta[a_l,b_l] < alpha :
                cluster[l] = np.array([a_l])
            else :
                index_a = np.where(Theta[a_l,:] >= alpha)
                index_b = np.where(Theta[b_l,:] >= alpha)
                cluster[l] = np.intersect1d(S,index_a,index_b)
        S = np.setdiff1d(S, cluster[l])
    
    return cluster


def perc_exact_recovery(O_hat, O_bar):
    value = 0
    for key1, true_clust in O_bar.items():
        for key2, est_clust in O_hat.items():
            if len(true_clust) == len(est_clust):
                test = np.intersect1d(true_clust,est_clust)
                if len(test) > 0 and len(test) == len(true_clust) and np.sum(np.sort(test) - np.sort(true_clust)) == 0 :
                    value +=1
    return value / len(O_bar)

d1 = 100
d2 = 100
n_sample = 900

copula1 = Logistic(n_sample = n_sample, d = d1, theta = 0.7)
copula2 = Logistic(n_sample = n_sample, d = d2, theta = 0.7)
sample1 = copula1.sample_unimargin()
sample2 = copula2.sample_unimargin()
sample = np.hstack([sample1, sample2])

train_sample, test_sample = train_test_split(sample, train_size = 1/3)

d = sample.shape[1]
R = np.zeros([train_sample.shape[0],d])
for j in range(0,d):
    X_vec = train_sample[:,j]
    R[:,j] = ecdf(X_vec)

Theta = np.ones([d,d])
for j in range(0,d):
    for i in range(0,j):
        Theta[i,j] = Theta[j,i] = 2 - theta(R[:,[i,j]])

cst = 4.0

alpha = cst * np.sqrt(np.log(d) / n_sample)

O_hat = clust(Theta, n = n_sample, alpha = alpha)

print(O_hat)

test_sample_1, test_sample_2 = train_test_split(test_sample, train_size = 1/2)
R_1 = np.zeros([test_sample_1.shape[0],d])
for j in range(0,d):
    X_vec = test_sample_1[:,j]
    R_1[:,j] = ecdf(X_vec)

Theta_1 = np.ones([d,d])
for j in range(0,d):
    for i in range(0,j):
        Theta[i,j] = Theta[j,i] = 2 - theta(R_1[:,[i,j]])

R_2 = np.zeros([test_sample_2.shape[0],d])
for j in range(0,d):
    X_vec = test_sample_2[:,j]
    R_2[:,j] = ecdf(X_vec)

Theta_1 = np.ones([d,d])
for j in range(0,d):
    for i in range(0,j):
        Theta[i,j] = Theta[j,i] = 2 - theta(R_2[:,[i,j]])

seco = SECO(R_1, R_2, O_hat)

print(seco)

O_bar = {1 : np.arange(0,d1), 2 : np.arange(d1,d1+d2)}

print(perc_exact_recovery(O_hat, O_bar))