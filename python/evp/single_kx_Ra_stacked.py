"""
Dedalus script for determining instability of static drizzle solutions to the Rainy-Benard system of equations.  This script computes curves of growth at discrete kx, scanning a variety of Rayleigh numbers.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves EVPs for an existing atmospheres, solved for by scripts in the nlbvp section.

Usage:
    convective_onset.py [options]

Options:
    --alpha=<alpha>   alpha value [default: 3]
    --beta=<beta>     beta value  [default: 1.1]
    --gamma=<gamma>   gamma value [default: 0.19]
    --q0=<q0>         basal q value [default: 0.6]

    --tau=<tau>       If set, override value of tau [default: 1e-3]
    --k=<k>           If set, override value of k [default: 1e4]

    --nondim=<n>      Non-Nondimensionalization [default: buoyancy]

    --Ra=<Ra>         Rayleigh number [default: 1e4]

    --min_kx=<mnkx>   Min kx [default: 0.1]
    --max_kx=<mxkx>   Max kx [default: 33]
    --num_kx=<nkx>    How many kxs to sample [default: 50]

    --top-stress-free     Stress-free upper boundary
    --stress-free         Stress-free both boundaries

    --nz=<nz>         Number of coeffs to use in eigenvalue search [default: 128]
    --target=<targ>   Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>     Target number of eigenvalues to search for [default: 20]

    --erf             Use an erf rather than a tanh for the phase transition
    --Legendre        Use Legendre polynomials

    --dense           Solve densely for all eigenvalues (slow)

    --plot_type=<plot_type>   File type for plots [default: pdf]
    --use-heaviside        Use the Heaviside function 
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
    logging.getLogger(system).setLevel(logging.WARNING)

import os
import numpy as np
import dedalus.public as de
import h5py
from pathlib import Path

from convective_onset_stacked import compute_growth_rate
from rainy_evp import SplitRainyBenardEVP, RainyBenardEVP, mode_reject
from etools import Eigenproblem
import matplotlib.pyplot as plt
plt.style.use('prl')
from docopt import docopt
args = docopt(__doc__)

Legendre = args['--Legendre']
erf = args['--erf']
nondim = args['--nondim']
plot_type = args['--plot_type']
use_heaviside = args['--use-heaviside']

N_evals = int(float(args['--eigs']))
target = float(args['--target'])

min_kx = float(args['--min_kx'])
max_kx = float(args['--max_kx'])
nkx = int(float(args['--num_kx']))
Ra = float(args['--Ra'])

if args['--stress-free']:
    bc_type = 'stress-free'
elif args['--top-stress-free']:
    bc_type = 'top-stress-free'
else:
    bc_type = None # default no-slip

dealias = 3/2
dtype = np.complex128

Prandtlm = 1
Prandtl = 1

Lz = 1
dealias = 2

α = float(args['--alpha'])
β = float(args['--beta'])
γ = float(args['--gamma'])
k = float(args['--k'])
q0 = float(args['--q0'])
tau = float(args['--tau'])
nz = int(float(args['--nz']))

logger.info('α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(α,β,γ,tau, k))

kxs = np.geomspace(min_kx, max_kx, num=nkx)
σ = []
for kx in kxs:
    σ_i = compute_growth_rate(kx, Ra, target=target, plot_type=plot_type, use_heaviside=use_heaviside)
    σ.append(σ_i)
    logger.info('Ra = {:.2g}, kx = {:.2g}, σ = {:.2g}'.format(Ra, kx, σ_i))
    if σ_i.imag > 0:
        # update target if on growing branch
        target = σ_i.imag
σ = np.array(σ)

# just for filename
if q0 == 1:
    EVP = RainyBenardEVP
else:
    EVP = SplitRainyBenardEVP

lo_res = EVP(nz, Ra, tau, kx, γ, α, β, q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside)
#lo_res.build_atmosphere()

# make plot
fig, ax = plt.subplots(figsize=[6,6/1.6])
ax.set_ylim(-0.5, 0.5)
ax.set_ylabel(r'$\omega_R$ (solid), $\omega_I$ (dashed)')

p = ax.semilogx(kxs, σ.real, label='Ra = {:.2g}'.format(Ra), color='k')
#ax.semilogx(kxs, σ.imag, linestyle='dashed',color='k')

fig_filename = f'growth_curve_Ra_{Ra}_nz_{nz:d}'
if args['--stress-free']:
    fig_filename += '_SF'
if args['--top-stress-free']:
    fig_filename += '_TSF'
if args['--dense']:
    fig_filename += '_dense'
plot_filename = Path(lo_res.case_name)/Path(f"{fig_filename}.pdf")
ax.axhline(0, color='k', linestyle=':',linewidth=1)
ax.legend()
omega_r_max_abs = np.abs(np.max(σ.real))
#ax.set_ylim(-1.1*omega_r_max_abs,1.1*omega_r_max_abs)
ax.set_xlim(0.1,100)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$\omega_r$")
plt.tight_layout()
ax.axhline(0, color='k', linestyle=':',linewidth=1)

fig.savefig(plot_filename)

