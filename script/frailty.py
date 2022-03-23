import numpy as np
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import sys
import Rmath

DBL_EPSILON = sys.float_info.epsilon


def generator(x):
    return 1-np.power(1-np.exp(-x), 1/theta)

theta = 1000
d = 50
n_sample = 1000

sample = np.zeros((n_sample, d))

#for i in range(0, n_sample):
#    V = np.random.gamma(1/theta, 1, 1)
#    E = np.random.gamma(1, 1, d)
#    _array = generator(E/V)
#    sample[i,:] = _array
#
#fig, ax = plt.subplots()
#ax.scatter(sample[:,0], sample[:,1], edgecolor = 'lightblue', color = 'white', s = 5)
#plt.savefig('/home/aboulin/CoPY/script/output/output.pdf')

def rSibuya(alpha, gamma_1_a):
    U = np.random.uniform(0.0,1.0,1)
    if (U <= alpha):
        return 1.0
    else:
        xMax = 1.0 / DBL_EPSILON
        Ginv = np.power((1-U)*gamma_1_a, -1.0/alpha)
        fGinv = math.floor(Ginv)
        if (Ginv > xMax) : 
            return fGinv
        if (1-U < 1.0 /(fGinv * Rmath.beta(fGinv, 1.0 - alpha))) :
            return math.ceil(Ginv)

        return fGinv

def rSibuya_vec(V, n, alpha):
    if (n >=1):
        gamma_1_a = Rmath.gammafn(1.0 - alpha)

        for i in range(0,n):
            V[i] = rSibuya(alpha, gamma_1_a)

def rSibuya_vec_c(n, alpha):
    res = np.zeros(n)
    rSibuya_vec(res, n, alpha)
    return res


for i in range(0, n_sample):
    V = rSibuya_vec_c(1, 1/theta)
    E = np.random.gamma(1,1,d)
    _array = generator(E/V)
    sample[i,:] = _array

print(sample)

fig, ax = plt.subplots()
ax.scatter(sample[:,0], sample[:,1], edgecolor = 'lightblue', color = 'white', s = 5)
plt.savefig('/home/aboulin/CoPY/script/output/output.pdf')

# code R

# X = rSibuya(n = 1000, alpha = 0.01)
# pdf('Documents/foobar.pdf)
# hist(log(X))
# dev.off()

"""
    Références

    Efficiently sampling nested Archimedeancopula
    Elements of Copula Modeling with R
"""