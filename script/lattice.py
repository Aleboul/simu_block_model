import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

xx, yy = np.meshgrid(np.arange(10), np.arange(10))

c_1 = np.repeat('blue', 40)
c_2 = np.repeat('red',60)

c = np.hstack((c_1,c_2))

fig, ax = plt.subplots()
ax.scatter(xx.flat, yy.flat, edgecolors= c, color = 'white', s = 10)

plt.savefig('lattice.pdf')