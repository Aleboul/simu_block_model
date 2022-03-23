import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

sys.path.insert(0, '/home/aboulin/')

import CoPY.copy.rng.utils as utils
import numpy as np

from CoPY.copy.rng.evd import Asy_neg_log

theta, psi1, psi2, d = 1/2, 0.5, 1.0, 2

copula = Asy_neg_log(theta = theta, psi1 = psi1, psi2 = psi2, d = d)

xlong = np.linspace(0.01,0.99,100)
output = []
for x in xlong:
    u = np.array([0.5, x])
    value = copula._dotC(u, 0)
    output.append(value)

print(output)