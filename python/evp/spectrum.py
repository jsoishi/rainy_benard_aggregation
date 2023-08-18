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
    --kx_min=<kx_min>      min x wavenumber [default: 0.1]
    --kx_max=<kx_max>      max x wavenumber [default: 10]
    --Lx=<Lx>              x box length. If specified, interpret kx_min,max as multiples of fundamental
    --n_kx=<n_kx>          number of x wavenumbers [default: 20]
    --top-stress-free      Stress-free upper boundary
    --stress-free          Stress-free both boundaries
 
    --nz=<nz>              Number of coeffs to use in eigenvalue search; if not set, uses resolution of background
    --target=<targ>        Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>          Target number of eigenvalues to search for [default: 20]
 
    --erf                  Use an erf rather than a tanh for the phase transition
    --Legendre             Use Legendre polynomials
 
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
plt.style.use("../../prl.mplstyle")

from rainy_evp import RainyBenardEVP
from etools import Eigenproblem
from docopt import docopt
args = docopt(__doc__)

N_evals = int(float(args['--eigs']))
target = float(args['--target'])
Rayleigh = float(args['--Ra'])
n_kx = int(args['--n_kx'])
tau_in = float(args['--tau'])
if args['--stress-free']:
    bc_type = 'stress-free'
elif args['--top-stress-free']:
    bc_type = 'top-stress-free'
else:
    bc_type = None # default no-slip
if args['--Lx']:
    Lx = float(args['--Lx'])
    logger.info(f'Lx = {Lx}. Ignoring kx_min and kx_max')
    kx_min = 2*np.pi/Lx
    kx_max = n_kx*kx_min
else:
    logger.info('Using kx_min and kx_max as wavenumbers')
    kx_min = float(args['--kx_min'])
    kx_max = float(args['--kx_max'])
    Lx = None

Prandtlm = 1
Prandtl = 1
dealias = 2

import os

def mode_reject(lo_res, hi_res):
    ep = Eigenproblem(None)
    ep.evalues_low   = lo_res.eigenvalues
    ep.evalues_high  = hi_res.eigenvalues
    evals_good, indx = ep.discard_spurious_eigenvalues()

    indx = np.argsort(evals_good.real)
    return evals_good, indx

if __name__ == "__main__":
    Legendre = args['--Legendre']
    erf = args['--erf']
    case = args['<case>']
    nondim = args['--nondim']
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
    if Lx:
        kxs = 2*np.pi/Lx * np.arange(n_kx)
    else:
        kxs = np.linspace(kx_min, kx_max, n_kx)
    # build solvers
    lo_res = RainyBenardEVP(nz, Rayleigh, tau, kxs[0], γ, α, β, lower_q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1)
    hi_res = RainyBenardEVP(3*nz/2, Rayleigh, tau, kxs[0], γ, α, β, lower_q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1)

    dlog = logging.getLogger('subsystems')
    dlog.setLevel(logging.WARNING)
    spectra = []
    fig = plt.figure(figsize=[6,6/2])
    fig_filename=f"Ra_{Rayleigh:.2e}_nz_{nz}_kx_min_{kx_min:.3f}_kx_max_{kx_max:.3f}_bc_{bc_type}_spectrum"
    re_ax = fig.add_axes([0.14,0.2,0.35,0.7])
    im_ax = fig.add_axes([0.64,0.2,0.35,0.7])
    max_growth = []
    for kx in kxs:
        for solver in [lo_res, hi_res]:
            if args['--dense']:
                solver.solve(Rayleigh, kx, dense=True)
            else:
                solver.solve(Rayleigh, kx, dense=False, N_evals=N_evals, target=target)
        evals_good, indx = mode_reject(lo_res, hi_res)
        max_growth.append(evals_good[indx][-1])
        eps = 1e-7
        col = np.where(np.abs(evals_good.imag) > eps, 'g',np.where(evals_good.real > 0, 'r','k'))
        re_ax.scatter(np.repeat(kx, evals_good.size),evals_good.real,marker='o', c=col)
        re_ax.set_prop_cycle(None)
        im_ax.scatter(np.repeat(kx, evals_good.size),evals_good.imag,marker='o', c=col)
        im_ax.set_prop_cycle(None)
    max_growth = np.array(max_growth)
    logger.info(f"maximum growth rate = {max_growth.max()}")
    re_ax.set_xlabel(r"$k_x$")
    re_ax.set_ylabel(r"$\Re{\sigma}$")

    im_ax.set_xlabel(r"$k_x$")
    im_ax.set_ylabel(r"$\Im{\sigma}$")

    spec_filename = lo_res.case_name+'/'+fig_filename+'.png'
    logger.info(f"saving file to {spec_filename}")
    fig.savefig(spec_filename, dpi=300)

