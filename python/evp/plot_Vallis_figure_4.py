import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd

Vallis_12 = pd.read_csv('Vallis_et_al_2019_data/beta_1.2.csv', names=['γ', 'Ra_c'])
Vallis_10 = pd.read_csv('Vallis_et_al_2019_data/beta_1.0.csv', names=['γ', 'Ra_c'])

fig, ax = plt.subplots()
f = h5py.File('critical_Ra_alpha3.0_beta1.0.h5', 'r')
ax.scatter(f['γ'][:], f['Ra_c'][:], label=r'd3, $\beta=1.0$', alpha=0.5)
f.close()
f = h5py.File('critical_Ra_alpha3.0_beta1.2.h5', 'r')
ax.scatter(f['γ'][:], f['Ra_c'][:], label=r'd3, $\beta=1.2$', alpha=0.5)
f.close()

ax.plot(Vallis_10['γ'], Vallis_10['Ra_c'], label=r'Vallis, $\beta=1.0$', linestyle='dashed', color='xkcd:dark green', marker='o')
ax.plot(Vallis_12['γ'], Vallis_12['Ra_c'], label=r'Vallis, $\beta=1.2$', linestyle='dashed', color='xkcd:dark red', marker='o')
ax.legend()

fig.savefig('Vallis_figure_4.png', dpi=300)
