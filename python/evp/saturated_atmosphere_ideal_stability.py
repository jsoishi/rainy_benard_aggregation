"""
Dedalus script for determining instability of static drizzle solutions to the Rainy-Benard system of equations.  This script computes curves of growth at discrete kx, scanning a variety of Rayleigh numbers.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''
(VPT19)

This script plots the ideal stability diagram for the saturated atmosphere, using Lambert W functions and an analytic solution.

Usage:
    saturated_atmosphere_ideal_stability.py [options]

Options:
    --alpha=<alpha>      Alpha parameter [default: 3]

    --VPT19_IVP          Use the VPT19 IVP atmosphere
    --unsaturated        Use an unsaturated atmosphere with q0=0.6

    --nz=<nz>            Vertical resolution [default: 128]
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)

import numpy as np
import dedalus.public as de
import h5py

from docopt import docopt
args = docopt(__doc__)

q_surface = 1

α = float(args['--alpha'])
nz = int(float(args['--nz']))

ΔT = -1
import analytic_atmosphere

dealias = 2
dtype = np.float64

Lz = 1
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)
dealias = 2
zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
z = zb.local_grid(1)
zd = zb.local_grid(dealias)

dz = lambda A: de.Differentiate(A, coords['z'])

nβ = 200
nγ = 100
grad_m = np.zeros((nβ, nγ))
grad_b = np.zeros((nβ, nγ))
βs = np.linspace(0, 2, num=nβ)
γs = np.linspace(0, 1, num=nγ)

for iβ, β in enumerate(βs):
    for iγ, γ in enumerate(γs):
        if args['--unsaturated']:
            case = 'unsaturated'
            sol = analytic_atmosphere.unsaturated
            if γ == 0.3:
                zc_analytic = 0.4832893544084419
                Tc_analytic = -0.4588071140209613
            elif γ == 0.19:
                zc_analytic = 0.4751621541611023
                Tc_analytic = -0.4588071140209616
            else:
                # hack
                zc_analytic = 0.4751621541611023
                Tc_analytic = -0.4588071140209616
            zc = zc_analytic
            Tc = Tc_analytic
            analytic_sol = sol(dist, zb, zc, Tc, β, γ, dealias=2, q0=0.6, α=3)
        elif args['--VPT19_IVP']:
            case = 'VPT19'
            sol = analytic_atmosphere.saturated_VPT19
            analytic_sol = sol(dist, zb, β, γ, dealias=2, q0=1, α=α)
        else:
            case = 'saturated'
            sol = analytic_atmosphere.saturated
            analytic_sol = sol(dist, zb, β, γ, dealias=2, q0=1, α=α)
        grad_m[iβ,iγ] = dz(analytic_sol['m']).evaluate()['g'][0,0,0]
        grad_b[iβ,iγ] = np.min(dz(analytic_sol['b']).evaluate()['g'][0,0,:]) # get min value

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}"

fig, ax = plt.subplots()
nlev = 11
b_mag = m_mag = (nlev-1)/2*0.3
grad_m_levels = np.linspace(-m_mag, m_mag, num=nlev)
grad_b_levels = np.linspace(-b_mag, b_mag, num=nlev)
csm = ax.contour(γs, βs, grad_m, grad_m_levels, colors='xkcd:dark blue')
ax.clabel(csm, csm.levels, fmt=fmt)
csb = ax.contour(γs, βs, grad_b, grad_b_levels, colors='xkcd:brick red')
ax.clabel(csb, csb.levels, fmt=fmt)
ax.contourf(γs, βs, (grad_m<0)&(grad_b>0), levels=[0.5, 1.5], colors='xkcd:dark green', alpha=0.5)
ax.set_title(r'$\alpha='+'{:}'.format(α)+r'$')
ax.set_ylabel(r'$\beta$')
ax.set_xlabel(r'$\gamma$')
ax.scatter(0.3, 1.1, alpha=0.5)
ax.scatter(0.3, 1.15, alpha=0.5)
ax.scatter(0.3, 1.2, alpha=0.5)
ax.scatter(0.3, 1.25, alpha=0.5)
ax.scatter(0.19, 1.1, marker='s', alpha=0.5)
ax.scatter(0.19, 1.15, marker='s', alpha=0.5)
ax.scatter(0.19, 1.2, marker='s', alpha=0.5)
ax.scatter(0.19, 1.25, marker='s', alpha=0.5)

fig.savefig('ideal_stability_alpha{:}_{:s}_figure_3.png'.format(α, case), dpi=300)
