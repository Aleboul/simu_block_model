import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-whitegrid')

seco = pd.read_csv('seco_model_1CV_ECO_1600.csv', index_col = 0)
perc = pd.read_csv('perc_model_1CV_ECO_1600.csv', index_col = 0)

print(seco)
print(perc)

mean_seco = seco.mean(axis = 0)
mean_perc = perc.mean(axis = 0)

print(mean_seco)
print(mean_perc)

mean_seco = np.log(1+mean_seco - np.min(mean_seco))

_alpha_ = np.array([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0])
fig, ax1 = plt.subplots()
ax1.plot(_alpha_, mean_perc, color = 'salmon', marker= 'd', linestyle = 'solid', markerfacecolor = 'white', lw = 1)
ax1.set_xlabel(r"$\frac{\alpha}{\sqrt{\ln(d)/n}}$")
ax1.set_ylabel("Exact recovery rate")
ax2 = ax1.twinx()
ax2.plot(_alpha_, mean_seco,color = 'lightblue', marker= 'd', linestyle = 'solid', markerfacecolor = 'white', lw = 1)
ax2.set_ylabel('SECO')
ax2.grid(None)
fig.savefig("exact_recov_rate.pdf")
