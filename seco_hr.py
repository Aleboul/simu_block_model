import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
plt.style.use('seaborn-whitegrid')

from coppy.rng.evd import Husler_Reiss


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

def wmado(R, w) :
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
        V[:,j] = np.power(R[:,j], 1/w[j])
    value_1 = np.amax(V,1)
    value_2 = (1/Nnb) * np.sum(V, 1)
    mado = (1/Tnb) * np.sum(value_1 - value_2)

    c = (1/Nnb)*np.sum(np.divide(w, 1 + np.array(w)))
    value = (mado + c) / (1-mado-c)
    return value

def crit(X, w, cols):
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

    clust = np.unique(cols)
    n = X.shape[0]
    d = X.shape[1]

    ### Compute rank
    R = np.zeros([n,d])
    for j in range(0,d):
        X_vec = np.array(X[:,j])
        R[:,j] = ecdf(X_vec)

    ### Evaluate the cluster as a whole

    value = wmado(R, w)

    _value_ = []
    for c in clust:
        index = np.where(cols == c)[0]
        _X = R[:,index]
        w_c = w[index]
        wei_clu = np.sum(w_c) / np.sum(w) 
        _value = wmado(_X, w_c / np.sum(w_c))
        _value_.append( wei_clu * _value)

    return -(value - np.sum(_value_))

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
    HR
"""

def init_pool_processes():
    sp.random.seed()

def operation_hr(dict, seed):
    """ Operation to perform Monte carlo simulation

    Input
    -----
        dict : dictionnary containing
                - d1 : dimension of the first sample
                - d2 : dimension of the second sample
                - n_sample : sample's length
    """
    sp.random.seed(seed * 5)
    values = []
    # Generate first sample
    Sigma = sym_matrix(d1, alea = True, a = 0.1, b=0.2) 
    Sigma = Sigma @ Sigma.T
    Gamma = Sigma2Gamma(Sigma)
    copula1 = Husler_Reiss(n_sample = n_sample, d = d1+1, Sigma = Gamma)
    sample1 = copula1.sample_unimargin()
    # Generate critnd sample
    Sigma = sym_matrix(d2, alea = True, a = 0.01, b=0.2) 
    Sigma = Sigma @ Sigma.T
    Gamma = Sigma2Gamma(Sigma)
    copula2 = Husler_Reiss(n_sample = n_sample, d = d2+1, Sigma = Gamma)
    sample2 = copula2.sample_unimargin()
    # merge sample
    sample = np.hstack((sample1, sample2))
    # initialization
    d = dict['d1'] + dict['d2'] +2
    w = np.repeat(1/(d), d)
    grp = np.arange(d)
    criteria = crit(sample, w, grp)
    values.append(criteria)
    grp[0] = -1
    for j in range(1,d):
        grp[j] = -1
        criteria = crit(sample, w,grp)
        if j >= d1+1:
            grp[j] = j
        values.append(criteria)

    return values

import multiprocessing as mp

d1 = 24
d2 = 74
n_sample = 200
n_iter = 200
pool = mp.Pool(processes= 8, initializer=init_pool_processes)

input = {'d1' : d1, 'd2': d2, 'n_sample' : n_sample}

result_objects = [pool.apply_async(operation_hr, args = (input,i)) for i in range(n_iter)]

results = [r.get() for r in result_objects]

pool.close()
pool.join()

df = pd.DataFrame(results)

print(df)

df.to_csv('CRIT_25_75_HR.csv')