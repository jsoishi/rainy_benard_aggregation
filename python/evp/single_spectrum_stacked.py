"""
Dedalus script for plotting spectrum of static drizzle solutions to the Rainy-Benard system of equations.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves EVPs for an existing atmospheres, solved for by scripts in the nlbvp section.

Usage:
    convective_onset.py <case> [options]

Options:
    <case>           Case (or cases) to calculate onset for

                           Properties of analytic atmosphere, if used
    --alpha=<alpha>        alpha value [default: 3]
    --beta=<beta>          beta value  [default: 1.1]
    --gamma=<gamma>        gamma value [default: 0.19]
    --q0=<q0>              basal q value [default: 0.6]

    --tau=<tau>            If set, override value of tau [default: 1e-3]
    --k=<k>                If set, override value of k [default: 1e4]

    --nondim=<n>           Non-Nondimensionalization [default: buoyancy]

    --Ra=<Ra>              Rayleigh number [default: 1e4]
    --kx=<kx>              x wavenumber [default: 0.1]
    --top-stress-free      Stress-free upper boundary
    --stress-free          Stress-free both boundaries

    --nz=<nz>              Number of coeffs to use in eigenvalue search; if not set, uses resolution of background [default: 128]
    --target=<targ>        Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>          Target number of eigenvalues to search for [default: 20]

    --erf                  Use an erf rather than a tanh for the phase transition
    --Legendre             Use Legendre polynomials
    --drift_threshold=<dt>        Drift threshold [default: 1e6]
    --relaxation_method=<re>      Method for relaxing the background [default: None]
    --rejection_method=<rej>      Method for rejecting modes [default: resolution]

    --dense                Solve densely for all eigenvalues (slow)
    --emode=<emode>        Index for eigenmode to visualize
    --annotate             Print mode indices on plot for identification
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
import matplotlib.pyplot as plt
#plt.style.use("../../prl.mplstyle")
plt.style.use("prl")

from rainy_evp import SplitRainyBenardEVP, RainyBenardEVP, mode_reject
from docopt import docopt
args = docopt(__doc__)

N_evals = int(float(args['--eigs']))
target = float(args['--target'])
Rayleigh = float(args['--Ra'])
tau_in = float(args['--tau'])
drift_threshold = float(args['--drift_threshold'])
if args['--stress-free']:
    bc_type = 'stress-free'
elif args['--top-stress-free']:
    bc_type = 'top-stress-free'
else:
    bc_type = None # default no-slip
kx = float(args['--kx'])
annotate = args['--annotate']
plot_type = args['--plot_type']
use_heaviside = args['--use-heaviside']
Prandtlm = 1
Prandtl = 1
dealias = 1#2

emode = args['--emode']
if emode:
    emode = int(emode)

import os

def evp_amp_reject(evp, indices, tol=1e-8):
    amp = []
    for i in indices:
        evp.solver.set_state(i,0)
        amp.append(np.max(np.abs(evp.fields['b']['g'])))
    amp = np.array(amp)
    indx = indices[np.where(amp > tol)]
    print(indx)
    print(indices)
    return evp.solver.eigenvalues[indx], indx

# def plot_eigenmode(evp, index, mode_label=None):
#     evp.solver.set_state(index,0)
#     fig, axes = plt.subplot_mosaic([['ux','.','uz'],
#                                     ['b', 'q','p']], layout='constrained')

#     z = evp.zb.local_grid(1).squeeze()
#     nz = z.shape[-1]
#     for v in ['b','q','p']:
#         if v == 'b':
#             i_max = np.argmax(np.abs(evp.fields[v]['g'].squeeze()))
#             phase_correction = evp.fields[v]['g'][0,...,i_max].squeeze()
#         evp.fields[v].change_scales(1)
#         name = evp.fields[v].name
#         data = evp.fields[v]['g'][0,...,:].squeeze()/phase_correction
#         axes[v].plot(data.real, z)
#         axes[v].plot(data.imag, z, ':')
#         axes[v].set_xlabel(f"${name}$")
#         axes[v].set_ylabel(r"$z$")
#     evp.fields['u'].change_scales(1)
#     u = evp.fields['u']['g']/phase_correction
#     axes['ux'].plot(u[0,0,...,:].squeeze().real, z)
#     axes['ux'].plot(u[0,0,...,:].squeeze().imag, z,':')
#     axes['ux'].set_xlabel(r"$u_x$")
#     axes['ux'].set_ylabel(r"$z$")
#     axes['uz'].plot(u[-1,0,...,:].squeeze().real, z)
#     axes['uz'].plot(u[-1,0,...,:].squeeze().imag, z, ':')
#     axes['uz'].set_xlabel(r"$u_z$")
#     axes['uz'].set_ylabel(r"$z$")
#     axes['q'].set_title(phase_correction)
#     sigma = evp.solver.eigenvalues[index]
#     fig.suptitle(f"$\sigma = {sigma.real:.3f} {sigma.imag:+.3e} i$")
#     if not mode_label:
#         mode_label = index
#     fig_filename=f"split_emode_indx_{mode_label}_Ra_{Rayleigh:.2e}_nz_{nz}_kx_{kx:.3f}_bc_{bc_type}"
#     fig.savefig(evp.case_name +'/'+fig_filename+'.png', dpi=300)
#     logger.info("eigenmode {:d} saved in {:s}".format(index, evp.case_name +'/'+fig_filename+'.png'))

if __name__ == "__main__":
    Legendre = args['--Legendre']
    erf = args['--erf']
    case = args['<case>']
    nondim = args['--nondim']
    relaxation_method = args['--relaxation_method']
    if case == 'analytic':
        α = float(args['--alpha'])
        β = float(args['--beta'])
        γ = float(args['--gamma'])
        k = float(args['--k'])
        lower_q0 = float(args['--q0'])
        tau = float(args['--tau'])
    else:
        raise NotImplementedError
        # f = h5py.File(case+'/drizzle_sol/drizzle_sol_s1.h5', 'r')
        # sol = {}
        # for task in f['tasks']:
        #     sol[task] = f['tasks'][task][0,0,0][:]
        # sol['z'] = f['tasks']['b'].dims[3][0][:]
        # tau_in = sol['tau'][0]
        # k = sol['k'][0]
        # α = sol['α'][0]
        # β = sol['β'][0]
        # γ = sol['γ'][0]
        # nz_sol = sol['z'].shape[0]
        # f.close()
    if args['--nz']:
        nz = int(float(args['--nz']))
    else:
        nz = nz_sol

    if args['--tau']:
        tau_in = float(args['--tau'])
    # build solvers
    if lower_q0 == 1:
        EVP = RainyBenardEVP
    else:
        EVP = SplitRainyBenardEVP
    lo_res = EVP(nz, Rayleigh, tau, kx, γ, α, β, lower_q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside)
    lo_res.plot_background()
    if args['--rejection_method'] == 'resolution':
        hi_res = EVP(int(2*nz), Rayleigh, tau, kx, γ, α, β, lower_q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside)
        hi_res.plot_background()
    elif args['--rejection_method'] == 'bases':
        hi_res = EVP(nz, Rayleigh, tau, kx, γ, α, β, lower_q0, k, Legendre=not(Legendre), erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside)
        hi_res.plot_background(label='alternative-basis')
    else:
        raise NotImplementedError('rejection method {:s}'.format(args['--rejection_method']))
    dlog = logging.getLogger('subsystems')
    dlog.setLevel(logging.WARNING)
    spectra = []
    fig = plt.figure(figsize=[12,6])
    spec_ax = fig.add_axes([0.15,0.2,0.8,0.7])
    for solver in [lo_res, hi_res]:
        if args['--dense']:
            sd_mode = 'dense'
            solver.solve(dense=True)
        else:
            sd_mode = 'sparse'
            solver.solve(dense=False, N_evals=N_evals, target=target)
    fig_filename=f"Ra_{Rayleigh:.2e}_nz_{nz}_kx_{kx:.3f}_bc_{bc_type}_{sd_mode}_spectrum"
    evals_ok, indx_ok, ep = mode_reject(lo_res, hi_res, plot_drift_ratios=True, drift_threshold=drift_threshold)
    evals_good = evals_ok
    indx = indx_ok
    #evals_good, indx = evp_amp_reject(lo_res, indx_ok)
    logger.info(f"good modes ({{$\delta_t$}} = {drift_threshold:.1e}):    max growth rate = {evals_good[-1]}")
    lo_indx = np.argsort(lo_res.eigenvalues.real)
    logger.info(f"low res modes: max growth rate = {lo_res.eigenvalues[lo_indx][-1]}")
    eps = 1e-7
    logger.info(f"good fastest oscillating modes: {evals_good[np.argmax(np.abs(evals_good.imag))]}")
    col = np.where(np.abs(evals_ok.imag) > eps, 'g', np.where(evals_ok.real > 0, 'r','k'))
    spec_ax.scatter(evals_ok.real, evals_ok.imag, marker='o', c=col, label=f'good modes ($\delta_t$ = {drift_threshold:.1e})',s=100)#, alpha=0.5, s=25)
    col = np.where(np.abs(evals_good.imag) > eps, 'g', np.where(evals_good.real > 0, 'r','k'))
    #spec_ax.scatter(evals_good.real, evals_good.imag, marker='s', c=col, label=f'good modes ($\delta_t$ = {drift_threshold:.1e}, $|q| > $ 1e-8)', alpha=0.5,zorder=0, s=100)
    #spec_ax.scatter(lo_res.eigenvalues.real, lo_res.eigenvalues.imag, marker='x', label='low res', alpha=0.4)
    #spec_ax.scatter(hi_res.eigenvalues.real, hi_res.eigenvalues.imag, marker='+', label='hi res', alpha=0.4)

    if annotate:
        for n,ev in enumerate(evals_ok):
            logger.info(f"ok ev = {ev}, index = {indx_ok[n]}")
            spec_ax.annotate(indx_ok[n], (ev.real, ev.imag), fontsize=8, ha='left',va='top')

        # for n,ev in enumerate(evals_good):
        #     logger.info(f"gd ev = {ev}, index = {indx[n]}")
        #     spec_ax.annotate(indx[n], (ev.real, ev.imag), fontsize=8, ha='left', va='bottom')
    #spec_ax.legend()
    # spec_ax.set_xlabel(r"$\Re{\sigma}$")
    # spec_ax.set_ylabel(r"$\Im{\sigma}$")
    spec_ax.set_xlabel(r"Growth Rate")
    spec_ax.set_ylabel(r"Frequency")
    spec_ax.set_xlim(-1,0.1)
    spec_ax.set_ylim(-0.3,0.3)
    spec_filename = f'{lo_res.case_name}/{fig_filename}.{plot_type}'
    logger.info(f"saving file to {spec_filename}")
    fig.tight_layout()
    fig.savefig(spec_filename, dpi=300)
    spec_ax.set_xlim(-0.2,0.5)
    spec_filename = f'{lo_res.case_name}/{fig_filename}_zoom.{plot_type}'
    logger.info(f"saving zoomed file to {spec_filename}")
    fig.tight_layout()
    fig.savefig(spec_filename, dpi=300)

    if emode is not None:
        lo_res.plot_eigenmode(emode, plot_type=plot_type)
    else:
        for i in [-1, -2, -3, -4, -5]:
            emode = indx[i]
            lo_res.plot_eigenmode(emode, plot_type=plot_type)
