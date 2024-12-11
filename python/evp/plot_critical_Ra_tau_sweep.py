"""
Script for plotting pre-computed critical Rayleigh numbers.

Usage:
    plot_critical_Ra <case>... [options]

Options:
    --q0=<q0>              Moisture at base of atmosphere [default: 1]
    --overlay_VPT19        Overlay points from VPT19
    --no_gamma_correction  Use original plots from VPT19, without gamma correction
"""
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("prl.mplstyle")

from docopt import docopt
args = docopt(__doc__)

data = {'γ':[],'β':[],'Ra':[],'k':[], 'σ':[], 'tau':[]}
for case in args['<case>']:
    print('opening {:}'.format(case))
    with h5py.File(case, 'r') as f:
        data['Ra'].append(f['crit/Ra'][()])
        data['k'].append(f['crit/k'][()])
        data['σ'].append(f['crit/σ'][()])
        data['β'].append(f['β'][()])
        data['γ'].append(f['γ'][()])
        data['tau'].append(f['tau'][()])
        f.close()
data = pd.DataFrame(data)
data.sort_values('tau', inplace=True)

ΔT = -1
α = 3
q0 = float(args['--q0'])
γ_crit = lambda β: -(β+ΔT)/(np.exp(α*ΔT)-q0)
β_crit = lambda γ: -γ*(np.exp(α*ΔT)-q0) - ΔT
our_m = lambda β, γ, z: γ + ((β+ΔT) + γ*(np.exp(α*ΔT)-1))*z

if q0 == 1:
    βs = [1.175, 1.1]
    label = 'saturated'
else:
    βs = [1, 1.05, 1.1]
    label = f'unsaturated_q{args["--q0"]}'
γs = [0.19]


# VPT19 conversions
M_convert = 1/3.8e-3
K2 = 4e-10
T0 = 5.5
VPT_γ_convert = G = (M_convert*K2*np.exp(α*T0))

fig, ax = plt.subplots(figsize=[8,8/1.6])
ax2 = ax.twinx()
for β in βs:
    curve = data[data['β']==β]
    Rac = curve['Ra'].min()
    p = ax.plot(curve['tau'], curve['Ra']/Rac, label=rf'$\beta = {β}$', marker='o', alpha=0.5)
    ax2.plot(curve['tau'], curve['k'], color=p[0].get_color(),
    linestyle='dashed', marker='s', alpha=0.5)
    #ax.axvline(x=γ_crit(β), color=p[0].get_color(), linestyle='dotted')
    print(f'β = {β}')
    print(curve[['γ', 'Ra','k','σ']])
ax.set_ylabel(r'$\mathrm{Ra}_c/\min({\mathrm{Ra}_c})$')
#ax.set_yscale('log')
ax2.set_ylabel('critical k')
ax2.set_ylim(1,3)
ax.set_xlabel(r'$\tau$')
ax.set_xscale('log')
ax.legend(fontsize='x-small', loc='center right')
xlim = ax.get_xlim()
ax.set_xlim(xlim[1],xlim[0])
fig.tight_layout()
fig.savefig(f'critical_Ra_and_k_vs_tau.png', dpi=300)
