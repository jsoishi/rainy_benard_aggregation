"""
Script for plotting zc and Tc for the Rainy-Benard equations.

Usage:
    plot_zc_Tc.py [options]

Options:
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
    logging.getLogger(system).setLevel(logging.WARNING)

import matplotlib.pyplot as plt
#plt.style.use("../prl.mplstyle")
plt.style.use("prl.mplstyle")

import numpy as np

from docopt import docopt
args = docopt(__doc__)

gamma_data = 'gamma_0.0_1.0_zc_Tc.csv'
q0_data = 'q0_0.15_0.99_zc_Tc.csv'

data = np.genfromtxt(gamma_data, delimiter=',')
γ = data[:,0]
zc = data[:,1]
Tc = data[:,2]

#fig, ax = plt.subplots(figsize=[8,8/1.6])
fig, ax = plt.subplots(figsize=[6,6/1.6])
ax.plot(γ, zc, label=r'$z_c$')
ax.plot(γ, -Tc, label=r'$-T_c$')
ax.set_xlabel(r'$\gamma$')
#ax.set_ylabel(r'$z_c$ and $-T_c$')
ax.set_ylabel('critical values')
ax.legend(title=r'$q_0 = 0.6$')
fig.tight_layout()
filename = 'zc_Tc_vs_gamma'
fig.savefig(filename +'.png', dpi=300)


data = np.genfromtxt(q0_data, delimiter=',')
q0 = data[:,0]
zc = data[:,1]
Tc = data[:,2]

#fig, ax = plt.subplots(figsize=[8,8/1.6])
fig, ax = plt.subplots(figsize=[6,6/1.6])
mask = (zc <=1) & (-Tc <= 1)
ax.plot(q0[mask], zc[mask], label=r'$z_c$')
ax.plot(q0[mask], -Tc[mask], label=r'$-T_c$')
q_min = q0[mask][0]
ax.plot((0,q_min), (1,1),
        color='xkcd:dark grey', linestyle='dashed', alpha=0.5)
ax.set_xlabel(r'$q_0$')
#ax.set_ylabel(r'$z_c$ and $-T_c$')
ax.set_ylabel('critical values')
ax.legend(title=r'$\gamma = 0.19$')
ax.set_xlim(0,1)
fig.tight_layout()
filename = 'zc_Tc_vs_q0'
fig.savefig(filename +'.png', dpi=300)

print(f'critical q_min = {q_min} ')
