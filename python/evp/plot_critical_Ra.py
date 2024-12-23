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

ΔT = -1
α = 3
q0 = float(args['--q0'])
γ_crit = lambda β: -(β+ΔT)/(np.exp(α*ΔT)-q0)
β_crit = lambda γ: -γ*(np.exp(α*ΔT)-q0) - ΔT
our_m = lambda β, γ, z: γ + ((β+ΔT) + γ*(np.exp(α*ΔT)-1))*z

if q0 == 1:
    βs = [1, 1.2]
    label = 'saturated'
else:
    βs = [1, 1.05, 1.1]
    label = f'unsaturated_q{args["--q0"]}'
γs = [0.19, 0.3]


# VPT19 conversions
M_convert = 1/3.8e-3
K2 = 4e-10
T0 = 5.5
VPT_γ_convert = G = (M_convert*K2*np.exp(α*T0))

fig, ax = plt.subplots(figsize=[8,8/1.6])
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
ax2.set_ylim(1,3)
ax.set_xlabel(r'$\beta$')
ax.legend(fontsize='x-small', loc='center left')
fig.tight_layout()
fig.savefig(f'critical_Ra_and_k_all_gamma_{label}.png', dpi=300)


fig, ax = plt.subplots(figsize=[8,8/1.6])
ax2 = ax.twinx()
for γ in γs:
    curve = data[data['γ']==γ]
    p = ax.plot(curve['β'], curve['Ra'], label=rf'$\gamma = {γ}$', marker='o', alpha=0.5)
    ax2.plot(curve['β'], curve['k'], color=p[0].get_color(),
    linestyle='dashed', marker='s', alpha=0.5)
    ax.axvline(x=β_crit(γ), color=p[0].get_color(), linestyle='dotted')
    print(f'γ = {γ}')
    print(curve[['β','Ra','k','σ']])
ax.set_ylabel('critical Ra')
ax.set_yscale('log')
ax2.set_ylabel('critical k')
ax2.set_ylim(1,3)
ax.set_xlabel(r'$\beta$')
ax.legend(fontsize='x-small', loc='center left')
fig.tight_layout()
fig.savefig(f'critical_Ra_and_k_gamma_{label}.png', dpi=300)

data.sort_values('γ', inplace=True)
fig, ax = plt.subplots(figsize=[8,8/1.6])
ax2 = ax.twinx()
for β in βs:
    curve = data[data['β']==β]
    p = ax.plot(curve['γ'], curve['Ra'], label=rf'$\beta = {β}$', marker='o', alpha=0.5)
    ax2.plot(curve['γ'], curve['k'], color=p[0].get_color(),
    linestyle='dashed', marker='s', alpha=0.5)
    ax.axvline(x=γ_crit(β), color=p[0].get_color(), linestyle='dotted')
    print(f'β = {β}')
    print(curve[['γ', 'Ra','k','σ']])
ax.set_ylabel('critical Ra')
ax.set_yscale('log')
ax2.set_ylabel('critical k')
ax2.set_ylim(1,3)
ax.set_xlabel(r'$\gamma$')
if args['--overlay_VPT19']:
    if args['--no_gamma_correction']:
        G = 1
    Vallis_12 = pd.read_csv('Vallis_et_al_2019_data/beta_1.2.csv', names=['γ', 'Ra_c'])
    Vallis_10 = pd.read_csv('Vallis_et_al_2019_data/beta_1.0.csv', names=['γ', 'Ra_c'])
    γ = Vallis_10['γ']*G
    ax.scatter(γ, 1/G*Vallis_10['Ra_c'],#*VPT_Ra_convert(1.0, γ),
               label=r'VPT19, $\beta=1.0$', color='xkcd:dark green', marker='o', alpha=0.5)
    γ = Vallis_12['γ']*G
    ax.scatter(γ, 1/G*Vallis_12['Ra_c'], #*VPT_Ra_convert(1.2, γ),
               label=r'VPT19, $\beta=1.2$', color='xkcd:dark red', marker='o', alpha=0.5)
ax.legend(fontsize='x-small', loc='center right')
fig.tight_layout()
fig.savefig(f'critical_Ra_and_k_beta_{label}.png', dpi=300)
print(VPT_γ_convert)

print('gamma crit')
for β in βs:
    print(f'β = {β:.2g}, γ_c = {γ_crit(β):.2g}')

print('beta crit')
for γ in γs:
    print(f'γ = {γ:.2g}, β_c = {β_crit(γ):.2g}')
