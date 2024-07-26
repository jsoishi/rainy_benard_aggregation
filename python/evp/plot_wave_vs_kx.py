"""
Plot waves vs k_x

Usage:
    plot_wave_vs_kx.py [options]

Options:
    --alpha=<alpha>        alpha value [default: 3]
    --beta=<beta>          beta value  [default: 1.1]
    --gamma=<gamma>        gamma value [default: 0.19]
    --q0=<q0>              basal q value [default: 0.6]

    --tau=<tau>            If set, override value of tau [default: 1e-3]
    --k=<k>                If set, override value of k [default: 1e4]

    --nondim=<n>           Non-Nondimensionalization [default: buoyancy]

    --Ra=<Ra>              Rayleigh number [default: 1e4]

    --eps=<eps>            wave threshold frequency [default: 1e-7]

    --min_kx=<mnkx>   Min kx [default: 0.1]
    --max_kx=<mxkx>   Max kx [default: 33]
    --num_kx=<nkx>    How many kxs to sample [default: 50]

    --top-stress-free      Stress-free upper boundary
    --stress-free          Stress-free both boundaries

    --nz=<nz>              Number of coeffs to use in eigenvalue search; if not set, uses resolution of background [default: 128]
    --target=<targ>        Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>          Target number of eigenvalues to search for [default: 20]
    --erf                  Use an erf rather than a tanh for the phase transition
    --Legendre             Use Legendre polynomials
    --drift_threshold=<dt>        Drift threshold [default: 1e6]
    --rejection_method=<rej>      Method for rejecting modes [default: resolution]

    --dense                Solve densely for all eigenvalues (slow)
    --plot_type=<plot_type>     File type for plots [default: pdf]
    --use-heaviside        Use the Heaviside function
    --restart              Don't solve, use saved eigenmodes and eigenvalues
"""
from docopt import docopt
args = docopt(__doc__)

import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
    logging.getLogger(system).setLevel(logging.WARNING)

import os
import numpy as np
import dedalus.public as de
import h5py
from pathlib import Path
from mpi4py import MPI

from rainy_evp import RainySpectrum
import matplotlib.pyplot as plt
plt.style.use('prl')

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    Legendre = args['--Legendre']
    erf = args['--erf']
    nondim = args['--nondim']
    plot_type = args['--plot_type']
    use_heaviside = args['--use-heaviside']
    dense = args['--dense']
    restart = args['--restart']

    rejection_method= args['--rejection_method']
    N_evals = int(float(args['--eigs']))
    target = float(args['--target'])

    min_kx = float(args['--min_kx'])
    max_kx = float(args['--max_kx'])
    nkx = int(float(args['--num_kx']))
    Rayleigh = float(args['--Ra'])
    eps = float(args['--eps'])

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
    coords = de.CartesianCoordinates('x', 'y', 'z')
    dist = de.Distributor(coords, dtype=dtype)
    dealias = 2

    α = float(args['--alpha'])
    β = float(args['--beta'])
    γ = float(args['--gamma'])
    k = float(args['--k'])
    q0 = float(args['--q0'])
    tau = float(args['--tau'])
    nz = int(float(args['--nz']))

    logger.info('α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(α,β,γ,tau, k))

    kxs_global = np.geomspace(min_kx, max_kx, num=nkx)
    kxs_local = kxs_global[comm.rank::comm.size]
    waves = []
    kx_plot = []
    for kx in kxs_local:
        spectrum = RainySpectrum(nz, Rayleigh, tau, kx, γ, α, β, q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside, restart=restart, rejection_method=rejection_method)
        spectrum.solve(dense=dense, N_evals=N_evals, target=target)

        wave_indx = np.abs(spectrum.evals_good.imag) > eps
        if wave_indx.sum() == 0:
            waves.append(np.array([0,]))
        else:
            waves.append(np.array(spectrum.evals_good[wave_indx]))
        kx_plot.append([kx,]*len(waves[-1]))
    global_waves = comm.gather(waves, root=0)
    global_kx_plot = comm.gather(kx_plot, root=0)

    for kxp,w in zip(global_kx_plot, global_waves):
        plt.scatter(kxp[-1], w[-1].imag, color='k')

    kz = 6.28
    N = np.sqrt(0.1)
    def gw_disp(kx, kz, N):
        return N*kx/np.sqrt((kx**2 + kz**2))
    plt.loglog(kxs_global, gw_disp(kxs_global,kz, N),'x-', label=r'$N_b \frac{kx}{\sqrt{kx^2 + (2\pi/Lz)^2}}$')
    plt.legend()
    #plt.ylim(-0.4,0.4)
    plt.ylim(1e-4,0.6)
    plt.xlabel(r"$k_x$")
    plt.ylabel(r"$\omega_i$")
    plt.tight_layout()

    fig_filename=f"Ra_{Rayleigh:.2e}_nz_{nz}_bc_{bc_type}_dense_{dense}_freq_vs_kx_parallel"
    spec_filename = f'{spectrum.lo_res.case_name}/{fig_filename}.{plot_type}'
    logger.info(f"saving file to {spec_filename}")


    plt.savefig(spec_filename,dpi=300)



