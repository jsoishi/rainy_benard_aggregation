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
    --saturated          Use saturated atmosphere (default)

    --mark_VPT19
    --no_mark

    --zoom

    --grad_b_max         Also mark maximum grad b

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

from analytic_zc import f_zc as zc_analytic
from analytic_zc import f_Tc as Tc_analytic

dealias = 2
dtype = np.float64

Lz = 1
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)
zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
z = zb.local_grid(1)
zd = zb.local_grid(dealias)

dz = lambda A: de.Differentiate(A, coords['z'])

nβ = 200
nγ = 100
grad_m = np.zeros((nβ, nγ))
grad_b = np.zeros((nβ, nγ))
grad_b_max = np.zeros((nβ, nγ))

if args['--zoom']:
    β0 = 0.95
    β1 = 1.2
    γ0 = 0.15
    γ1 = 0.35
else:
    β0 = 0
    β1 = 2
    γ0 = 0
    γ1 = 1

βs = np.linspace(β0, β1, num=nβ)
γs = np.linspace(γ0, γ1, num=nγ)

for iβ, β in enumerate(βs):
    for iγ, γ in enumerate(γs):
        if args['--unsaturated']:
            case = 'unsaturated'
            sol = analytic_atmosphere.unsaturated

            zc = zc_analytic()
            Tc = Tc_analytic()
            analytic_sol = sol(dist, zb, β, γ, zc(γ), Tc(γ), dealias=2, q0=0.6, α=3)
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
        grad_b_max[iβ,iγ] = np.max(dz(analytic_sol['b']).evaluate()['g'][0,0,:])
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}"

fig, ax = plt.subplots(figsize=[4,4/1.6])
nlev = 17
b_mag = m_mag = (nlev-1)/2*0.3
grad_m_levels = np.linspace(-m_mag, m_mag, num=nlev)
grad_b_levels = np.linspace(-b_mag, b_mag, num=nlev)
csm = ax.contour(γs, βs, grad_m, grad_m_levels, colors='xkcd:dark blue')
ax.clabel(csm, csm.levels, fmt=fmt)
csb = ax.contour(γs, βs, grad_b, grad_b_levels, colors='xkcd:brick red')
ax.clabel(csb, csb.levels, fmt=fmt)
if args['--grad_b_max']:
    csbm = ax.contour(γs, βs, grad_b_max, grad_b_levels, colors='xkcd:forest green')
    ax.clabel(csbm, csbm.levels, fmt=fmt)

ax.contourf(γs, βs, (grad_m<0)&(grad_b>0), levels=[0.5, 1.5], colors='xkcd:dark green', alpha=0.25)
ax.contourf(γs, βs, (grad_m>0)&(grad_b>0), levels=[0.5, 1.5], colors='xkcd:grey', alpha=0.25)
ax.set_title(r'$\alpha='+'{:}'.format(α)+r'$')
ax.set_ylabel(r'$\beta$')
ax.set_xlabel(r'$\gamma$')
if args['--mark_VPT19']:
    ax.scatter(0.19, 1.2, marker='*', alpha=0.5, s=100)
elif not args['--no_mark']:
    ax.scatter(0.3, 1.15, alpha=0.5)
    ax.scatter(0.19, 1., marker='s', alpha=0.5)
    ax.scatter(0.19, 1.05, marker='s', alpha=0.5)
    ax.scatter(0.19, 1.1, marker='s', alpha=0.5)
    ax.scatter(0.19, 1.15, marker='s', alpha=0.5)
    ax.scatter(0.19, 1.175, marker='*', alpha=0.5)
    ax.plot(0.19, 1.1, marker='.', color='black')
fig.tight_layout()

filename = 'ideal_stability_alpha{:}_{:s}_figure_3'.format(α, case)
if args['--zoom']:
    filename += '_zoom'
fig.savefig(filename +'.png', dpi=300)
