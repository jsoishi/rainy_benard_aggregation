"""
Dedalus script for determining instability of static drizzle solutions to the Rainy-Benard system of equations.  This script computes curves of growth at discrete kx, scanning a variety of Rayleigh numbers.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves EVPs for an existing atmospheres, solved for by scripts in the nlbvp section.

Usage:
    convective_onset.py [options]

Options:
                      Properties of analytic atmosphere, if used
    --alpha=<alpha>   alpha value [default: 3]
    --beta=<beta>     beta value  [default: 1.1]
    --gamma=<gamma>   gamma value [default: 0.19]
    --q0=<q0>         basal q value [default: 0.6]

    --tau=<tau>       If set, override value of tau
    --k=<k>           If set, override value of k

    --nondim=<n>      Non-Nondimensionalization [default: buoyancy]

    --min_Ra=<minR>   Minimum Rayleigh number to sample [default: 1e4]
    --max_Ra=<maxR>   Maximum Rayleigh number to sample [default: 1e5]
    --num_Ra=<nRa>    How many Rayleigh numbers to sample [default: 5]

    --top-stress-free     Stress-free upper boundary
    --stress-free         Stress-free both boundaries

    --nz=<nz>         Number of coeffs to use in eigenvalue search; if not set, uses resolution of background
    --target=<targ>   Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>     Target number of eigenvalues to search for [default: 20]

    --erf             Use an erf rather than a tanh for the phase transition
    --Legendre        Use Legendre polynomials

    --relaxation_method=<re>     Method for relaxing the analytic atmosphere

    --dense           Solve densely for all eigenvalues (slow)

    --verbose         Show plots on screen
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)

import os
import numpy as np
import dedalus.public as de
import h5py

from rainy_evp import RainyBenardEVP
from etools import Eigenproblem

import matplotlib.pyplot as plt

from docopt import docopt
args = docopt(__doc__)

Legendre = args['--Legendre']
erf = args['--erf']
nondim = args['--nondim']
relaxation_method = args['--relaxation_method']

N_evals = int(float(args['--eigs']))
target = float(args['--target'])

if args['--stress-free']:
    bc_type = 'stress-free'
elif args['--top-stress-free']:
    bc_type = 'top-stress-free'
else:
    bc_type = None # default no-slip


dealias = 3/2
dtype = np.complex128

def mode_reject(lo_res, hi_res):
    ep = Eigenproblem(None,use_ordinal=False, drift_threshold=1e6)
    ep.evalues_low   = lo_res.eigenvalues
    ep.evalues_high  = hi_res.eigenvalues
    evals_good, indx = ep.discard_spurious_eigenvalues()

    # fig, ax = plt.subplots()
    # ep.plot_drift_ratios(axes=ax)
    # fig.savefig('./drift_ratios.png', dpi=300)
    indx = np.argsort(evals_good.real)
    return evals_good, indx


Prandtlm = 1
Prandtl = 1

Lz = 1
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)
dealias = 2

α = float(args['--alpha'])
β = float(args['--beta'])
γ = float(args['--gamma'])
k = float(args['--k'])
q0 = float(args['--q0'])
tau = float(args['--tau'])

case = 'analytic'
relaxation_method = 'none'

nz = int(float(args['--nz']))

logger.info('α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(α,β,γ,tau, k))

# def plot_eigenfunctions(σ):
#     i_max = np.argmax(np.abs(b['g'][0,0,:]))
#     phase_correction = b['g'][0,0,i_max]
#     u['g'][:] /= phase_correction
#     b['g'] /= phase_correction
#     q['g'] /= phase_correction
#     fig, ax = plt.subplots(figsize=[6,6/1.6])
#     for Q in [u, q, b]:
#         if Q.tensorsig:
#             for i in range(3):
#                 p = ax.plot(Q['g'][i][0,0,:].real, z[0,0,:], label=Q.name+r'$_'+'{:s}'.format(coords.names[i])+r'$')
#                 ax.plot(Q['g'][i][0,0,:].imag, z[0,0,:], linestyle='dashed', color=p[0].get_color())
#         else:
#             p = ax.plot(Q['g'][0,0,:].real, z[0,0,:], label=Q)
#             ax.plot(Q['g'][0,0,:].imag, z[0,0,:], linestyle='dashed', color=p[0].get_color())
#     ax.set_title(r'$\omega_R = ${:.3g}'.format(σ.real)+ r' $\omega_I = ${:.3g}'.format(σ.imag)+' at kx = {:.3g} and Ra = {:.3g}'.format(kx['g'][0,0,0].real, Rayleigh['g'][0,0,0].real))
#     ax.legend()
#     fig_filename = 'eigenfunctions_{:}_Ra{:.2g}_kx{:.2g}_nz{:d}'.format(nondim, Rayleigh['g'][0,0,0].real, kx['g'][0,0,0].real, nz)
#     fig.savefig(case+'/'+fig_filename+'.png', dpi=300)

# fix Ra, find omega
def compute_growth_rate(kx, Ra, low_res, high_res, target=0):
    for solver in [lo_res, hi_res]:
        if args['--dense']:
            solver.solve(Ra, kx, dense=True)
        else:
            solver.solve(Ra, kx, dense=False, N_evals=N_evals, target=target)
    evals_good, indx = mode_reject(lo_res, hi_res)

    i_evals = np.argsort(evals_good.real)
    evals = evals_good[i_evals]
    peak_eval = evals[-1]
    # choose convention: return the positive complex mode of the pair
    if peak_eval.imag < 0:
        peak_eval = np.conj(peak_eval)

    return peak_eval

def peak_growth_rate(*args):
    rate = compute_growth_rate(*args)
    # flip sign so minimize finds maximum
    return -1*rate.real

growth_rates = {}
Ras = np.geomspace(float(args['--min_Ra']),float(args['--max_Ra']),num=int(float(args['--num_Ra'])))
kxs = np.logspace(-1, 1.5, num=50)
print(Ras)
for Rayleigh in Ras:
    σ = []
    # reset to base target for each Ra loop
    target = float(args['--target'])
    kx = kxs[0]
    lo_res = RainyBenardEVP(nz, Rayleigh, tau, kx, γ, α, β, q0, k, relaxation_method=relaxation_method, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1)
    lo_res.plot_background()
    hi_res = RainyBenardEVP(int(3*nz/2), Rayleigh, tau, kx, γ, α, β, q0, k, relaxation_method=relaxation_method, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1)
    hi_res.plot_background()

    for kx in kxs:
        σ_i = compute_growth_rate(kx, Rayleigh, lo_res, hi_res, target=target)
        σ.append(σ_i)
        logger.info('Ra = {:.2g}, kx = {:.2g}, σ = {:.2g}'.format(Rayleigh, kx, σ_i))
        if σ_i.imag > 0:
            # update target if on growing branch
            target = σ_i.imag
    growth_rates[Rayleigh] = np.array(σ)

fig, ax = plt.subplots(figsize=[6,6/1.6])
peak_σ = -np.inf

if nondim == 'diffusion':
    ax2 = ax.twinx()
    ax.set_ylim(-15, 25)
    ax2.set_ylim(1e-1, 1e3)
    ax2.set_yscale('log')
    ax.set_ylabel(r'$\omega_R$ (solid)')
    ax2.set_ylabel(r'$\omega_I$ (dashed)')
elif nondim == 'buoyancy':
    ax2 = ax
    ax.set_ylim(-0.05, 0.025)
    ax.set_ylabel(r'$\omega_R$ (solid), $\omega_I$ (dashed)')

for Ra in growth_rates:
    σ = growth_rates[Ra]
    peak_σ = max(peak_σ, np.max(σ))
    p = ax.plot(kxs, σ.real, label='Ra = {:.2g}'.format(Ra))
    ax2.plot(kxs, σ.imag, linestyle='dashed', color=p[0].get_color())
ax.set_xscale('log')

fig_filename = 'growth_curves_{:}_nz{:d}'.format(nondim, nz)
if args['--stress-free']:
    fig_filename += '_SF'
if args['--top-stress-free']:
    fig_filename += '_TSF'
if args['--dense']:
    fig_filename += '_dense'
ax.legend()
ax.axhline(y=0, linestyle='dashed', color='xkcd:grey', alpha=0.5)
ax.set_title(r'$\gamma$ = {:}, $\beta$ = {:}, $\tau$ = {:}'.format(γ,β,tau))
ax.set_xlabel('$k_x$')
ax.set_title('{:} timescales'.format(nondim))
fig.savefig(lo_res.case_name+'/'+fig_filename+'.png', dpi=300)



import scipy.optimize as sciop
bounds = sciop.Bounds(lb=1, ub=10)

peaks = {'σ':[], 'k':[], 'Ra':[]}
for Ra in growth_rates:
    σ = growth_rates[Ra]
    peak_i = np.argmax(σ)
    kx = kxs[peak_i]
    lo_res = RainyBenardEVP(nz, Ra, tau, kx, γ, α, β, q0, k, relaxation_method=relaxation_method, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1)
    hi_res = RainyBenardEVP(int(3*nz/2), Ra, tau, kx, γ, α, β, q0, k, relaxation_method=relaxation_method, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1)

    result = sciop.minimize(peak_growth_rate, kx, args=(Ra, lo_res, hi_res), bounds=bounds, method='Nelder-Mead', tol=1e-5)
    # obtain full complex rate

    σ = compute_growth_rate(result.x[0], Ra, lo_res, hi_res)
    logger.info('peak search: start at Ra = {:.4g}, kx = {:.4g}, found σ_max = {:.2g},{:.2g}i, kx = {:.4g}'.format(Ra, kx, σ.real, σ.imag, result.x[0]))
    peaks['σ'].append(σ)
    peaks['k'].append(result.x[0])
    peaks['Ra'].append(Ra)
    #plot_eigenfunctions(σ)

for key in ['σ', 'k', 'Ra']:
    peaks[key] = np.array(peaks[key])

from scipy.interpolate import interp1d
f_σR_i = interp1d(peaks['σ'].real, peaks['k']) #inverse
f_σR = interp1d(peaks['k'], peaks['σ'].real)
f_σI_i = interp1d(peaks['σ'].imag, peaks['k']) #inverse
f_σI = interp1d(peaks['k'], peaks['σ'].imag)

# to find critical Ra
f_σR_Ra_i = interp1d(peaks['σ'].real, peaks['Ra'])
f_σ_Ra = interp1d(peaks['Ra'], peaks['σ'])
f_k_Ra = interp1d(peaks['Ra'], peaks['k'])

peak_ks = np.geomspace(np.min(peaks['k']), np.max(peaks['k']))

crit_Ra = f_σR_Ra_i(0)
crit_k = f_k_Ra(crit_Ra)
crit_σ = f_σ_Ra(crit_Ra)
logger.info('Critical point, based on interpolation:')
logger.info('Ra = {:}, k = {:}'.format(crit_Ra, crit_k))
logger.info('σ = {:}, {:}i'.format(crit_σ.real, crit_σ.imag))

lo_res = RainyBenardEVP(nz, crit_Ra, tau, crit_k, γ, α, β, q0, k, relaxation_method=relaxation_method, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1)
hi_res = RainyBenardEVP(int(3*nz/2), crit_Ra, tau, crit_k, γ, α, β, q0, k, relaxation_method=relaxation_method, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1)
σ = compute_growth_rate(crit_k, crit_Ra, lo_res, hi_res)
#plot_eigenfunctions(σ)
logger.info('σ = {:}, {:}i'.format(crit_σ.real, crit_σ.imag))
logger.info('σ = {:}, {:}i (sigma)'.format(σ.real, σ.imag))

ax.plot(peak_ks, f_σR(peak_ks), linestyle='dotted', color='xkcd:grey')
ax.scatter(crit_k, crit_σ.real, color='xkcd:grey', marker='x')
ax.scatter(crit_k, σ.real, color='xkcd:pink', marker='o', alpha=0.5)

fig_filename = 'growth_curves_peaks_{:}_nz{:d}'.format(nondim, nz)
if args['--stress-free']:
    fig_filename += '_SF'
if args['--top-stress-free']:
    fig_filename += '_TSF'
if args['--dense']:
    fig_filename += '_dense'
fig.savefig(lo_res.case_name+'/'+fig_filename+'.png', dpi=300)
