import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py

plt.style.use('../../prl.mplstyle')
t_avg_start = 1000
gamma = 0.19

def calc_n_Cneg(Q_eq):
    neg = Q_eq < 0
    zneg = neg.sum(axis=-1)
    nz = neg.shape[-1]
    return zneg/nz
b_profile = {}
q_profile = {}
m_profile = {}
z_profile = {}
fig = plt.figure(figsize=(8,6),layout='constrained')
axd = fig.subplot_mosaic(
"""
bqm.
tttt
"""
)

# b_ax = zfig.add_subplot(131)
# q_ax = zfig.add_subplot(132)
# m_ax = zfig.add_subplot(133)

for kexp in range(3,7):
    filebase = f"analytic_unsaturated/alpha3_beta1.1_gamma0.19/tau0.1_k1e{kexp}_erf_Legendre/rainy_benard_Ra1e10_tau0.1_k1e+0{kexp}_nz128_nx512/"
    snaps  = filebase + "snapshots/snapshots_virt_joined.h5"
    traces = filebase + "traces/traces_s1.h5"
    with h5py.File(snaps,'r') as df:
        b_profile[kexp] = df['tasks/b_avg'][t_avg_start:,0,0,0:].mean(axis=0)
        q_profile[kexp] = df['tasks/q_avg'][t_avg_start:,0,0,0:].mean(axis=0)
        m_profile[kexp] = df['tasks/m_avg'][t_avg_start:,0,0,0:].mean(axis=0)
        z_profile[kexp] = df['tasks/b_avg'].dims[3][0][:]
        #n_Cneg_profile = calc_n_Cneg(df['tasks/Q_eq'][:])[t_avg_start:,0,0,0:].mean(axis=0)
    with h5py.File(traces,'r') as df:
        Re_trace = df['tasks/Re'][:,0,0,0]
        time = df['scales/sim_time'][:]
    axd['b'].plot(b_profile[kexp], z_profile[kexp], label=f'$k = 10^{kexp:d}$')
    axd['q'].plot(q_profile[kexp], z_profile[kexp])#, label=f'$k = 10^{kexp:d}$')
    axd['m'].plot(m_profile[kexp], z_profile[kexp])#, label=f'$k = 10^{kexp:d}$')
    axd['t'].plot(time, Re_trace)
    calc_m = b_profile[kexp] + gamma*q_profile[kexp]
    print(f"L2(calc_m, m) = {np.linalg.norm(calc_m-m_profile[kexp])}")
    
for ax in ['b', 'q', 'm']:
    axd[ax].set_ylabel(r"$z$")
    axd[ax].set_ylim(0,1)
axd['b'].set_xlabel(r"$\left< b \right>$")
axd['q'].set_xlabel(r"$\left< q \right>$")
axd['m'].set_xlabel(r"$\left< m \right>$")
axd['m'].xaxis.set_ticks([0.11,0.113])
axd['t'].set_xlabel(r"$t/t_b$")
axd['t'].set_ylabel(r"$\mathrm{Re}$")
axd['t'].set_xlim(0,50000)
fig.legend(bbox_to_anchor=(1.,0.9))
#fig.tight_layout()
fig.savefig("b_z_ksweep_test.pdf",bbox_inches="tight")
    

