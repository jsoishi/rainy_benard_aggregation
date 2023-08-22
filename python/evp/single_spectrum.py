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

    --tau=<tau>            If set, override value of tau
    --k=<k>                If set, override value of k

    --nondim=<n>           Non-Nondimensionalization [default: buoyancy]

    --Ra=<Ra>              Minimum Rayleigh number to sample [default: 1e4]
    --kx=<kx>              x wavenumber [default: 0.1]
    --top-stress-free      Stress-free upper boundary
    --stress-free          Stress-free both boundaries

    --nz=<nz>              Number of coeffs to use in eigenvalue search; if not set, uses resolution of background
    --target=<targ>        Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>          Target number of eigenvalues to search for [default: 20]

    --erf                  Use an erf rather than a tanh for the phase transition
    --Legendre             Use Legendre polynomials
    --drift_threshold=<dt>        Drift threshold [default: 1e3]
    --relaxation_method=<re>      Method for relaxing the background [default: IVP]

    --dense                Solve densely for all eigenvalues (slow)
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

from rainy_evp import RainyBenardEVP, mode_reject
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

Prandtlm = 1
Prandtl = 1
dealias = 2

import os

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
    lo_res = RainyBenardEVP(nz, Rayleigh, tau, kx, γ, α, β, lower_q0, k, relaxation_method=relaxation_method, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1)
    lo_res.plot_background()
    hi_res = RainyBenardEVP(int(3*nz/2), Rayleigh, tau, kx, γ, α, β, lower_q0, k, relaxation_method=relaxation_method, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1)
    hi_res.plot_background()
    dlog = logging.getLogger('subsystems')
    dlog.setLevel(logging.WARNING)
    spectra = []
    fig = plt.figure(figsize=[6,6/2])
    fig_filename=f"Ra_{Rayleigh:.2e}_nz_{nz}_kx_{kx:.3f}_bc_{bc_type}_spectrum"
    spec_ax = fig.add_axes([0.15,0.2,0.8,0.7])
    for solver in [lo_res, hi_res]:
        if args['--dense']:
            solver.solve(Rayleigh, kx, dense=True)
        else:
            solver.solve(Rayleigh, kx, dense=False, N_evals=N_evals, target=target)
    evals_good, indx,ep = mode_reject(lo_res, hi_res, plot_drift=True, drift_threshold=drift_threshold)
    logger.info(f"good modes ($\delta_t$ = {drift_threshold:.1e}):    max growth rate = {evals_good[-1]}")
    lo_indx = np.argsort(lo_res.eigenvalues.real)
    logger.info(f"low res modes: max growth rate = {lo_res.eigenvalues[lo_indx][-1]}")
    eps = 1e-7
    logger.info(f"good fastest oscillating modes: {evals_good[np.argmax(np.abs(evals_good.imag))]}")
    col = np.where(np.abs(evals_good.imag) > eps, 'g', np.where(evals_good.real > 0, 'r','k'))
    spec_ax.scatter(evals_good.real, evals_good.imag, marker='o', c=col, label=f'good modes ($\delta_t$ = {drift_threshold:.1e})')
    spec_ax.scatter(lo_res.eigenvalues.real, lo_res.eigenvalues.imag, marker='x', label='low res', alpha=0.4)
    spec_ax.scatter(hi_res.eigenvalues.real, hi_res.eigenvalues.imag, marker='+', label='hi res', alpha=0.4)
    spec_ax.legend()
    spec_ax.set_xlabel(r"$\Re{\sigma}$")
    spec_ax.set_ylabel(r"$\Im{\sigma}$")
    spec_ax.set_xlim(-10,1)
    spec_filename = lo_res.case_name+'/'+fig_filename+'.png'
    logger.info(f"saving file to {spec_filename}")
    fig.tight_layout()
    fig.savefig(spec_filename, dpi=300)
    spec_ax.set_xlim(-0.2,0.5)
    spec_filename = lo_res.case_name+'/'+fig_filename+'_zoom.png'
    logger.info(f"saving zoomed file to {spec_filename}")
    fig.tight_layout()
    fig.savefig(spec_filename, dpi=300)
