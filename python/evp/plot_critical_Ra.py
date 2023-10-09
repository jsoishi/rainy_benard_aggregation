"""
Script for plotting pre-computed critical Rayleigh numbers.

Usage:
    plot_critical_Ra <case>... [options]

Options:
    --overlay_VPT19        Overlay points from VPT19
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
    print(curve[['Ra','k','σ']])
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
    print(f'γ = {γ}')
    print(curve[['β','Ra','k','σ']])
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
    print(f'β = {β}')
    print(curve[['γ', 'Ra','k','σ']])
ax.set_ylabel('critical Ra')
ax.set_yscale('log')
ax2.set_ylabel('critical k')
ax.set_xlabel(r'$\gamma$')
if args['--overlay_VPT19']:
    Vallis_12 = pd.read_csv('Vallis_et_al_2019_data/beta_1.2.csv', names=['γ', 'Ra_c'])
    Vallis_10 = pd.read_csv('Vallis_et_al_2019_data/beta_1.0.csv', names=['γ', 'Ra_c'])
    ax.scatter(Vallis_10['γ'], Vallis_10['Ra_c'], label=r'Vallis, $\beta=1.0$', color='xkcd:dark green', marker='o', alpha=0.5)
    ax.scatter(Vallis_12['γ'], Vallis_12['Ra_c'], label=r'Vallis, $\beta=1.2$', color='xkcd:dark red', marker='o', alpha=0.5)

ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig('critical_Ra_and_k_beta.png', dpi=300)
