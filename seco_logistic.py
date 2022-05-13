import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
plt.style.use('seaborn-whitegrid')

from coppy.rng.evd import Logistic


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

def init_pool_processes():
    sp.random.seed()

def operation_logistic(dict, seed):
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
    copula1 = Logistic(n_sample = dict['n_sample'], d = dict['d1'], theta = np.random.uniform(0.3,0.8,1))
    sample1 = copula1.sample_unimargin()
    # Generate second sample
    copula2 = Logistic(n_sample = dict['n_sample'], d = dict['d2'], theta = np.random.uniform(0.3,0.8,1))
    sample2 = copula2.sample_unimargin()
    # Merge both samples to obtain one
    sample = np.hstack((sample1,sample2))
    # initialization
    d = dict['d1'] + dict['d2']
    w = np.repeat(1/d,d)
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

input = {'d1' : d1+1, 'd2': d2+1, 'n_sample' : n_sample}

result_objects = [pool.apply_async(operation_logistic, args = (input,i)) for i in range(n_iter)]

results = [r.get() for r in result_objects]

pool.close()
pool.join()

print(results)
df = pd.DataFrame(results)

df.to_csv('CRIT_25_75_LOGISTIC.csv')