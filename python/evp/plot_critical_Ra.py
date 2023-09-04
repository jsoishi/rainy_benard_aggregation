"""
Script for plotting pre-computed critical Rayleigh numbers.

Usage:
    plot_critical_Ra <case>...
"""
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from docopt import docopt
args = docopt(__doc__)

data = {'γ':[],'β':[],'Ra':[],'k':[], 'σ':[]}
for case in args['<case>']:
    print('opening {:}'.format(case))
    with h5py.File(case, 'r') as f:
        data['Ra'].append(f['crit/Ra'][()])
        data['k'].append(f['crit/k'][()])
        data['σ'].append(f['crit/σ'][()])
        data['β'].append(f['β'][()])
        data['γ'].append(f['γ'][()])
        f.close()
data = pd.DataFrame(data)
data.sort_values('β', inplace=True)

fig, ax = plt.subplots(figsize=[6,6/1.6])
ax2 = ax.twinx()
for γ in np.sort(data['γ'].unique()):
    curve = data[data['γ']==γ]
    p = ax.plot(curve['β'], curve['Ra'], label=rf'$\gamma = {γ}$', marker='o', alpha=0.5)
    ax2.plot(curve['β'], curve['k'], color=p[0].get_color(),
    linestyle='dashed', marker='s', alpha=0.5)
    print(curve['σ'])
ax.set_ylabel('critical Ra')
ax.set_yscale('log')
ax2.set_ylabel('critical k')
ax.set_xlabel(r'$\beta$')
ax.legend(loc='center left')
fig.tight_layout()
fig.savefig('critical_Ra_and_k_all_gamma.png', dpi=300)


fig, ax = plt.subplots(figsize=[6,6/1.6])
ax2 = ax.twinx()
for γ in [0.19, 0.3]:
    curve = data[data['γ']==γ]
    p = ax.plot(curve['β'], curve['Ra'], label=rf'$\gamma = {γ}$', marker='o', alpha=0.5)
    ax2.plot(curve['β'], curve['k'], color=p[0].get_color(),
    linestyle='dashed', marker='s', alpha=0.5)
    print(curve['σ'])
ax.set_ylabel('critical Ra')
ax.set_yscale('log')
ax2.set_ylabel('critical k')
ax.set_xlabel(r'$\beta$')
ax.legend(loc='center left')
fig.tight_layout()
fig.savefig('critical_Ra_and_k_gamma.png', dpi=300)

data.sort_values('γ', inplace=True)
fig, ax = plt.subplots(figsize=[6,6/1.6])
ax2 = ax.twinx()
for β in [1, 1.2]:
    curve = data[data['β']==β]
    p = ax.plot(curve['γ'], curve['Ra'], label=rf'$\beta = {β}$', marker='o', alpha=0.5)
    ax2.plot(curve['γ'], curve['k'], color=p[0].get_color(),
    linestyle='dashed', marker='s', alpha=0.5)
    print(curve['σ'])
ax.set_ylabel('critical Ra')
ax.set_yscale('log')
ax2.set_ylabel('critical k')
ax.set_xlabel(r'$\gamma$')
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig('critical_Ra_and_k_beta.png', dpi=300)
