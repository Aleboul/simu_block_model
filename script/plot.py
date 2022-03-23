import sys
from time import process_time

sys.path.insert(0, '/home/aboulin/')

import numpy as np
import math
import matplotlib.pyplot as plt

from CoPY.src.rng.evd import Bilog

import seaborn as sns
from scipy.stats import norm
plt.style.use('seaborn-whitegrid')

np.random.seed(42)

theta = [0.1,0.8,1/3]
n_sample = 1000

copula = Bilog(n_sample = n_sample, d = 3, theta = theta)
sample = copula.sample_unimargin()
print(sample)
#asy = [0.4,0.1,0.6,[0.3,0.2], [0.1,0.1], [0.4,0.1], [0.2,0.3,0.2]]
#theta = 1.0
#psi1 = 0.2
#psi2 = 1.0
#Sigma = (2*norm.ppf(0.52))**2 * np.matrix([[0,1,1],[1,0,1],[1,1,0]])
#n_sample = 1024
#copula = tEV(Sigma = Sigma, n_sample = n_sample, d = 3, psi1 = psi1)
#sample = copula.sample_unimargin()
crest = sns.color_palette('crest', as_cmap = True)
#print(sample)

fig, ax = plt.subplots()
ax.scatter(sample[:,0], sample[:,1], edgecolor = crest(0.6), color = 'white', s = 5)

#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter3D(sample[:,0],sample[:,1],sample[:,2], s = 5.0, edgecolor = crest(0.5), color = 'white')
#ax.set_xlabel(r'$u_0$')
#ax.set_ylabel(r'$u_1$')
#ax.set_zlabel(r'$u_2$')
#
plt.savefig('/home/aboulin/CoPY/script/output/hr.pdf')