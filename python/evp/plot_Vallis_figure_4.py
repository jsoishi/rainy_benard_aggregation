import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd

Vallis_12 = pd.read_csv('Vallis_et_al_2019_data/beta_1.2.csv', names=['γ', 'Ra_c'])

f = h5py.File('critical_Ra_alpha3.0_beta1.2.h5', 'r')

fig, ax = plt.subplots()
ax.scatter(f['γ'][:], f['Ra_c'][:], label='d3')
ax.scatter(Vallis_12['γ'], Vallis_12['Ra_c'], label='Vallis')
ax.legend()

fig.savefig('Vallis_figure_4.png', dpi=300)
print(f['γ'][:])
print(f['Ra_c'][:])
