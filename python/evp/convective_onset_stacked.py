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

    --tau=<tau>       If set, override value of tau [default: 1e-3]
    --k=<k>           If set, override value of k   [default: 1e4]

    --nondim=<n>      Non-Nondimensionalization [default: buoyancy]

    --min_Ra=<minR>   Minimum Rayleigh number to sample [default: 1e4]
    --max_Ra=<maxR>   Maximum Rayleigh number to sample [default: 1e5]
    --num_Ra=<nRa>    How many Rayleigh numbers to sample [default: 5]
    --Ra_list=<ra_list>    List of Rayleigh numbers to sample [default: None]

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

    --tol_crit_Ra=<tol>    Tolerance on frequency for critical growth [default: 1e-5]
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

if args['--stress-free']:
    bc_type = 'stress-free'
elif args['--top-stress-free']:
    bc_type = 'top-stress-free'
else:
    bc_type = None # default no-slip


dealias = 1
dtype = np.complex128

Prandtlm = 1
Prandtl = 1

Lz = 1
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)

α = float(args['--alpha'])
β = float(args['--beta'])
γ = float(args['--gamma'])
k = float(args['--k'])
q0 = float(args['--q0'])
tau = float(args['--tau'])

nz = int(float(args['--nz']))

logger.info('α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(α,β,γ,tau, k))

def compute_growth_rate(kx, Ra, target=0, plot_fastest_mode=False, plot_type='png', use_heaviside=False):
    if q0 == 1:
        EVP = RainyBenardEVP
    else:
        EVP = SplitRainyBenardEVP
    lo_res = EVP(nz, Ra, tau, kx, γ, α, β, q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside)
    hi_res = EVP(int(3*nz/2), Ra, tau, kx, γ, α, β, q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside)
    for solver in [lo_res, hi_res]:
        if args['--dense']:
            solver.solve(dense=True)
        else:
            solver.solve(dense=False, N_evals=N_evals, target=target)
    evals_good, indx, ep = mode_reject(lo_res, hi_res, plot_drift_ratios=False)

    i_evals = np.argsort(evals_good.real)
    evals = evals_good[i_evals]
    peak_eval = evals[-1]
    # choose convention: return the positive complex mode of the pair
    if peak_eval.imag < 0:
        peak_eval = np.conj(peak_eval)

    if plot_fastest_mode:
        lo_res.plot_eigenmode(indx[-1], plot_type=plot_type)
    return peak_eval

def peak_growth_rate(*args):
    rate = compute_growth_rate(*args)
    # flip sign so minimize finds maximum
    return -1*rate.real

def find_continous_peak(Ra, kx, plot_fastest_mode=False, plot_type='png',use_heaviside=False):

    result = sciop.minimize(peak_growth_rate, kx, args=(Ra), bounds=bounds, method='Nelder-Mead', tol=1e-5)
    # obtain full complex rate
    σ = compute_growth_rate(result.x[0], Ra, plot_fastest_mode=plot_fastest_mode, plot_type=plot_type, use_heaviside=use_heaviside)
    return result.x[0], σ

if __name__ == "__main__":
    growth_rates = {}
    ra_list = args['--Ra_list']
    if ra_list != 'None':
        Ras = [float(r) for r in ra_list.split(',')]
        print("Ra_list = ", ra_list)
        print("number of Ras = ",len(Ras))
    else:
        Ras = np.geomspace(float(args['--min_Ra']),float(args['--max_Ra']),num=int(float(args['--num_Ra'])))
    kxs = np.geomspace(min_kx, max_kx, num=nkx)
    print(Ras)
    for Ra in Ras:
        σ = []
        # reset to base target for each Ra loop
        target = float(args['--target'])
        kx = kxs[0]
        if q0 == 1:
            EVP = RainyBenardEVP
        else:
            EVP = SplitRainyBenardEVP
        lo_res = EVP(nz, Ra, tau, kx, γ, α, β, q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside)
        # lo_res.plot_background(plot_type=plot_type)
        # hi_res = EVP(int(3*nz/2), Ra, tau, kx, γ, α, β, q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside)
        # hi_res.plot_background(plot_type=plot_type)
        for system in ['rainy_evp']:
             logging.getLogger(system).setLevel(logging.WARNING)

        for kx in kxs:
            σ_i = compute_growth_rate(kx, Ra, target=target, plot_type=plot_type, use_heaviside=use_heaviside)
            σ.append(σ_i)
            logger.info('Ra = {:.2g}, kx = {:.2g}, σ = {:.2g}'.format(Ra, kx, σ_i))
            if σ_i.imag > 0:
                # update target if on growing branch
                target = σ_i.imag
        σ = np.array(σ)
        growth_rates[Ra] = {'σ':σ, 'max σ.real':σ[np.argmax(σ.real)]}

    fig, ax = plt.subplots(figsize=[6,6/1.6])

    if nondim == 'diffusion':
        ax2 = ax.twinx()
        ax.set_ylim(-15, 25)
        ax2.set_ylim(1e-1, 1e3)
        ax2.set_yscale('log')
        ax.set_ylabel(r'$\omega_R$ (solid)')
        ax2.set_ylabel(r'$\omega_I$ (dashed)')
    elif nondim == 'buoyancy':
        ax2 = ax
        ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel(r'$\omega_R$ (solid), $\omega_I$ (dashed)')

    for Ra in growth_rates:
        σ = growth_rates[Ra]['σ']
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
    ax.set_ylim(-0.1,0.1)
    ax.set_title('{:} timescales'.format(nondim))
    plt.tight_layout()
    fig.savefig(lo_res.case_name+'/'+fig_filename+'.png', dpi=300)

    for system in ['rainy_evp']:
         logging.getLogger(system).setLevel(logging.WARNING)

    import scipy.optimize as sciop
    bounds = sciop.Bounds(lb=1, ub=10)
    # find Ra bracket
    σ_re = -np.inf
    lower_Ra = None
    upper_Ra = None
    for Ra in growth_rates:
        σ_re = growth_rates[Ra]['max σ.real']
        if σ_re < 0:
            lower_Ra = Ra
        else:
            upper_Ra = Ra
            break
    if not lower_Ra or not upper_Ra:
        raise ValueError("Sampled Rayleigh numbers do not bound instability (lower: {:}, upper: {:})".format(lower_Ra, upper_Ra))

    # find peak growth rates of bracket
    peaks = {}
    for Ra in [lower_Ra, upper_Ra]:
        σ = growth_rates[Ra]['σ']
        peak_i = np.argmax(σ.real)
        kx0 = kxs[peak_i] # initial guess
        kx, σ = find_continous_peak(Ra, kx0, plot_type=plot_type)
        peaks[Ra] = {'σ':σ, 'k':kx}

    # conduct a bracketing search with interpolation to find critical Ra
    σ = np.inf
    tol = float(args['--tol_crit_Ra'])
    iter = 0
    max_iter = 15
    while np.abs(σ.real) > tol and iter < max_iter:
        σs = np.array([peaks[lower_Ra]['σ'], peaks[upper_Ra]['σ']])
        ks = np.array([peaks[lower_Ra]['k'], peaks[upper_Ra]['k']])
        ln_Ras = np.log(np.array([lower_Ra, upper_Ra]))
        # do this in log Ra
        ln_crit_Ra = np.interp(0, σs.real, ln_Ras)
        crit_k = np.interp(ln_crit_Ra, ln_Ras, ks)
        crit_σ = np.interp(ln_crit_Ra, ln_Ras, σs)
        crit_Ra = np.exp(ln_crit_Ra)

        logger.info('Critical point, based on interpolation:')
        logger.info('Ra = {:.3g}, k = {:}'.format(crit_Ra, crit_k))

        kx, σ = find_continous_peak(crit_Ra, crit_k, plot_fastest_mode=True, use_heaviside=use_heaviside)
        logger.info('σ = {:.2g}, {:.2g}i (calculated) at k = {:}'.format(σ.real, σ.imag, kx))

        if σ.real > 0:
            upper_Ra = crit_Ra
        else:
            lower_Ra = crit_Ra
        if not crit_Ra in peaks:
            peaks[crit_Ra] = {'σ':σ, 'k':kx}
        crit_k = kx
        iter+=1

    logger.info("critical Ra found after {:d} iterations".format(iter))

    for Ra in peaks:
        ax.scatter(peaks[Ra]['k'], peaks[Ra]['σ'].real, color='xkcd:grey', marker='x', alpha=0.5)

    ax.scatter(crit_k, σ.real, color='xkcd:pink', marker='o', alpha=0.5)

    fig_filename = 'growth_curves_peaks_{:}_nz{:d}'.format(nondim, nz)
    if args['--stress-free']:
        fig_filename += '_SF'
    if args['--top-stress-free']:
        fig_filename += '_TSF'
    if args['--dense']:
        fig_filename += '_dense'
    fig_filename = f"{lo_res.case_name}/{fig_filename}.{plot_type}"
    fig.savefig(fig_filename, dpi=300)
    logger.info("peaks plotted in {:}".format(fig_filename))

    f_curves = lo_res.case_name+'/critical_curves_nz_{:d}.h5'.format(nz)
    with h5py.File(f_curves,'w') as f:
        f['α'] = α
        f['β'] = β
        f['γ'] = γ
        f['q0'] = q0
        f['tau'] = tau
        f['k_q'] = k
        f['nz'] = nz
        # critical value
        f['crit/Ra'] = crit_Ra
        f['crit/k'] = crit_k
        f['crit/σ'] = peaks[crit_Ra]['σ']
        f['crit/iter'] = iter
        f['crit/max_iter'] = max_iter
        f['crit/tol'] = tol
        for i, Ra in enumerate(peaks):
            f[f'peaks/{i:d}/Ra'] = Ra
            f[f'peaks/{i:d}/k'] = peaks[Ra]['k']
            f[f'peaks/{i:d}/σ'] = peaks[Ra]['σ']
        for Ra in growth_rates:
            f[f'curves/{Ra:.4e}/Ra'] = Ra
            f[f'curves/{Ra:.4e}/k'] = kxs
            f[f'curves/{Ra:.4e}/σ'] = growth_rates[Ra]['σ']
        f.close()
    logger.info("Critical curves written out to: {:}".format(f_curves))
