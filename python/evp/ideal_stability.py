"""
Dedalus script for determining instability of static drizzle solutions to the Rainy-Benard system of equations.  This script computes curves of growth at discrete kx, scanning a variety of Rayleigh numbers.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''
(VPT19)

This script plots the ideal stability diagram for the saturated atmosphere, using Lambert W functions and an analytic solution.

Usage:
    ideal_stability.py <atmosphere> [options]

where <atmosphere> is one of:
    VPT19              Use the VPT19 IVP atmosphere
    unsaturated        Use an unsaturated atmosphere with q0=0.6
    saturated          Use saturated atmosphere (default)

Options:
    --alpha=<alpha>      Alpha parameter [default: 3]

    --no_mark            Suppress all markings

    --zoom               Zoom on an atmosphere-specific subsection of the range
    --one_contour

    --grad_b_max         Also mark maximum grad b

    --low_res

    --nz=<nz>            Vertical resolution [default: 128]
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
    logging.getLogger(system).setLevel(logging.WARNING)

import matplotlib.pyplot as plt
#plt.style.use("../../prl.mplstyle")
plt.style.use("prl.mplstyle")

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

dealias = 1
dtype = np.float64

Lz = 1
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)
zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
z = dist.local_grid(zb)
zd = dist.local_grid(zb, scale=dealias)

dz = lambda A: de.Differentiate(A, coords['z'])

if args['--low_res']:
    nβ = 20
    nγ = 10
else:
    nβ = 200
    nγ = 100

grad_m = np.zeros((nβ, nγ))
grad_b = np.zeros((nβ, nγ))
grad_b_max = np.zeros((nβ, nγ))

ε=0.622
e0=611 #Pa
p0=1e5 #Pa
K2 = 4e-10
T0 = 5.5
qc = ε*e0/p0
G = K2*np.exp(α*T0)/qc

case = args['<atmosphere>']
if args['<atmosphere>'] == 'saturated':
    cases       = [(0.19, 1.2), (0.19, 1.175), (0.19, 1.1), (0.19, 1.0)]
    cases_VPT19 = [(0.19*G, 1.2)]
    if args['--zoom']:
         β0 = 0.975
         β1 = 1.275
    label_stable = (0.25, 1.2575)
    label_moist = (0.25, 1.2125)
    label_unstable = (0.25, 1.125)
    label_rot = 18 # measured via Krita, 16deg for grad_b_min, 20deg for grad_m
elif args['<atmosphere>'] == 'unsaturated':
    cases = [(0.19, 1.15), (0.19, 1.1), (0.19, 1.05), (0.19, 1.0)]
    cases_VPT19 = [(None, None)]
    if args['--zoom']:
         β0 = 0.95
         β1 = 1.2
    label_stable = (0.25, 1.175)
    label_moist = (0.25, 1.1125)
    label_unstable = (0.25, 1.0)
    label_rot = 10
elif args['<atmosphere>'] == 'VPT19':
    cases_2d = [(0.19, 1.2)]
    cases_3d = [(None, None)]
    if args['--zoom']:
         β0 = 0.95
         β1 = 1.3
else:
    raise ValueError("<atmosphere>]{:s} not in valid set of choices".format(args['<atmosphere>']))

if args['--zoom']:
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
        if case == 'unsaturated':
            sol = analytic_atmosphere.unsaturated
            zc = zc_analytic()
            Tc = Tc_analytic()
            analytic_sol = sol(dist, zb, β, γ, zc(γ), Tc(γ), dealias=1, q0=0.6, α=3)
        elif case == 'saturated':
            sol = analytic_atmosphere.saturated
            analytic_sol = sol(dist, zb, β, γ, dealias=1, q0=1, α=α)
        elif case == 'VPT19':
            sol = analytic_atmosphere.saturated_VPT19
            analytic_sol = sol(dist, zb, β, γ, dealias=1, q0=1, α=α)

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
def fmt_tick(x):
    s = f"{x:.3f}"
    if s.endswith("0"):
        s = f"{x:.2f}"
    return rf"{s}"
def fmt_m(x):
    return r"$\nabla m = "+rf"{fmt(x)}"+r"$"
def fmt_b_min(x):
    return r"$\nabla b_\mathrm{min} = "+rf"{fmt(x)}"+r"$"
def fmt_b_max(x):
    return r"$\nabla b_\mathrm{max} = "+rf"{fmt(x)}"+r"$"

fig, ax = plt.subplots(figsize=[8,8/1.6])
if args['--one_contour']:
    nlev = 1
    mag = 0
elif args['--zoom']:
    nlev = 7
    mag = 0.05
else:
    nlev = 17
    mag = 0.3
b_mag = m_mag = (nlev-1)/2*mag
grad_m_levels = np.linspace(-m_mag, m_mag, num=nlev)
grad_b_levels = np.linspace(-b_mag, b_mag, num=nlev)
csm = ax.contour(γs, βs, grad_m, grad_m_levels, colors='xkcd:dark blue')
csb = ax.contour(γs, βs, grad_b, grad_b_levels, colors='xkcd:brick red')
if args['--grad_b_max']:
    csbm = ax.contour(γs, βs, grad_b_max, grad_b_levels, colors='xkcd:forest green')
    ax.clabel(csbm, csbm.levels, fmt=fmt_b_max)

ax.contourf(γs, βs, (grad_m<0)&(grad_b>0), levels=[0.5, 1.5], colors='xkcd:grey', alpha=0.5)
ax.contourf(γs, βs, (grad_m>0)&(grad_b>0), levels=[0.5, 1.5], colors='xkcd:dark grey', alpha=0.75)
#ax.set_title(r'$\alpha='+'{:}'.format(α)+r'$')
ax.set_ylabel(r'$\beta$')
ax.set_xlabel(r'$\gamma$')
if not args['--no_mark']:
    color = 'xkcd:forest green'
    for gamma, beta in cases:
        ax.scatter(gamma, beta, marker='s', alpha=0.5, color=color)
    for gamma, beta in cases_VPT19:
        ax.scatter(gamma, beta, marker='^', alpha=0.5, color=color)
    ax.text(*label_stable, 'stable',
            horizontalalignment='center', verticalalignment='center', rotation=0)
    ax.text(*label_moist, 'dry stable, moist unstable',
            horizontalalignment='center', verticalalignment='center', rotation=label_rot)
    ax.text(*label_unstable, 'unstable',
            horizontalalignment='center', verticalalignment='center', rotation=0)
ax.clabel(csm, csm.levels, fmt=fmt_m, fontsize='small')
ax.clabel(csb, csb.levels, fmt=fmt_b_min, fontsize='small')
if args['--zoom']:
    # xticks = np.linspace(γ0, γ1, num=9)
    xticks = ax.get_xticks()
    xlabels = [fmt_tick(x) for x in xticks]
    ax.set_xticks(xticks, labels=xlabels)
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)

fig.tight_layout()

filename = 'ideal_stability_{:s}_alpha{:}'.format(case, α)
if args['--zoom']:
    filename += '_zoom'
fig.savefig(filename +'.png', dpi=300)
