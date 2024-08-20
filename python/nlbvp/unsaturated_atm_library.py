"""
Dedalus script for solving static drizzle solutions to the Rainy-Benard system of equations.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves NLBVPs for unsaturated atmospheres, which corresponds to Vallis et al 2019, Figure 2.


Usage:
    unsaturated_atm.py [options]

Options:
    --q0=<q0>            Relative humidity at lower boundary [default: 0.6]
    --alpha=<alpha>      Alpha parameter [default: 3]
    --gamma=<gamma>      Gamma parameter [default: 0.3]
    --beta=<beta>        Beta parameter  [default: 1.2]

    --start_tau=<st>     Starting value for tau sweep [default: 1e-3]
    --end_tau=<et>       Ending value for tau sweep [default: 1e-6]
    --num_tau=<nt>       Number of taus to sample in tau sweep [default: 7]

    --reverse_search_ks  Reverse direction of k search (from high to low)

    --Legendre

    --erf                Use erf, not tanh

    --tolerance=<t>      Tolerance for convergence [default: 1e-5]
    --niter=<n>          Max iterations before stopping [default: 50]
    --damping=<d>        Damping rate for newton iterations [default: 0.9]

    --nz=<nz>            Vertical (z) grid resolution [default: 256]
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)
import matplotlib.pyplot as plt

import numpy as np
from dedalus import public as de

from fractions import Fraction
import time
import os

from docopt import docopt
args = docopt(__doc__)

q_surface = float(args['--q0'])
nz = int(args['--nz'])

α = float(args['--alpha'])
β = float(args['--beta'])
γ = float(args['--gamma'])

ΔT = -1

start_tau = float(args['--start_tau'])
stop_tau = float(args['--end_tau'])
num_tau = int(float(args['--num_tau']))
taus = np.geomspace(start_tau, stop_tau, num=num_tau)
ks = np.logspace(3, 5, num=11)

Prandtl = 1
Prandtlm = 1
P = 1                                 #  diffusion on buoyancy
S = (Prandtlm/Prandtl)**(-1/2)        #  diffusion on moisture

data_dir = 'unsaturated_atm_alpha{:}_beta{:}_gamma{:}_q{:}'.format(args['--alpha'], args['--beta'], args['--gamma'], args['--q0'])

if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

tol = float(args['--tolerance'])

Lz = 1

# Create bases and domain
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=np.float64)
dealias = 2
if args['--Legendre']:
    zb = de.Legendre(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
else:
    zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
z = dist.local_grid(zb)
zd = dist.local_grid(zb, scale=dealias)

b = dist.Field(name='b', bases=zb)
q = dist.Field(name='q', bases=zb)

τb1 = dist.Field(name='τb1')
τb2 = dist.Field(name='τb2')
τq1 = dist.Field(name='τq1')
τq2 = dist.Field(name='τq2')

lift_basis = zb.derivative_basis(2)
lift = lambda A, n: de.Lift(A, lift_basis, n)

ex, ey, ez = coords.unit_vector_fields(dist)

k = dist.Field(name='k')

from scipy.special import erf
if args['--erf']:
    # go for maximal supression, rather than equivalent width
    H = lambda A: 0.5*(1+erf(k*A))
else:
    H = lambda A: 0.5*(1+np.tanh(k*A))

z_grid = dist.Field(name='z_grid', bases=zb)
z_grid['g'] = z

temp = b - β*z_grid
temp.name = 'T'

qs = np.exp(α*temp)
rh = q*np.exp(-α*temp)

tau = dist.Field(name='tau')

α_f = dist.Field()
α_f['g'] = α
β_f = dist.Field()
β_f['g'] = β
γ_f = dist.Field()
γ_f['g'] = γ

dt = lambda A: 0*A

from scipy.special import lambertw as W
def compute_analytic(z_in, zc, Tc):
    q = dist.Field(bases=zb)
    T = dist.Field(bases=zb)
    q.change_scales(dealias)
    T.change_scales(dealias)

    z = dist.Field(bases=zb)
    z.change_scales(dealias)
    z['g'] = z_in

    b1 = 0
    b2 = β + ΔT
    q1 = q_surface
    q2 = np.exp(α*ΔT)

    bc = Tc + β*zc
    qc = np.exp(α*Tc)

    P = bc + γ*qc
    Q = ((b2-bc) + γ*(q2-qc))

    C = P + Q*(z['g']-zc)/(1-zc) - β*z['g']

    mask = (z['g']>=zc)
    T['g'][mask] = C[mask] - W(α*γ*np.exp(α*C[mask])).real/α
    T['g'][~mask] = Tc*z['g'][~mask]/zc

    b = (T + β*z).evaluate()
    q['g'][mask] = np.exp(α*T['g'][mask])
    q['g'][~mask] = q1 + (qc-q1)*z['g'][~mask]/zc
    m = (b + γ*q).evaluate()
    rh = (q*np.exp(-α*T)).evaluate()
    return {'b':b, 'q':q, 'm':m, 'T':T, 'rh':rh, 'z':z, 'γ':γ}


from analytic_zc import f_zc as zc_analytic
from analytic_zc import f_Tc as Tc_analytic
analytic = compute_analytic(zd, zc_analytic()(γ), Tc_analytic()(γ))

def plot_solution(z, solution, title=None, mask=None, linestyle=None, ax=None):
    b = solution['b']['g']
    q = solution['q']['g']
    m = solution['m']['g']
    T = solution['T']['g']
    rh = solution['rh']['g']
    γ = solution['γ']

    if mask is None:
        mask = np.ones_like(z, dtype=bool)
    if ax is None:
        fig, ax = plt.subplots(ncols=2, sharey=True)
        markup = True
        return_fig = True
    else:
        for axi in ax:
            axi.set_prop_cycle(None)
        markup = False
        return_fig = False

    ax[0].plot(b[mask],z[mask], label='$b$', linestyle=linestyle)
    ax[0].plot(γ*q[mask],z[mask], label='$\gamma q$', linestyle=linestyle)
    ax[0].plot(m[mask],z[mask], label='$b+\gamma q$', linestyle=linestyle)

    ax[1].plot(T[mask],z[mask], label='$T$', linestyle=linestyle)
    ax[1].plot(q[mask],z[mask], label='$q$', linestyle=linestyle)
    ax[1].plot(rh[mask],z[mask], label='$r_h$', linestyle=linestyle)
    ax[1].axvline(x=1, linestyle='dotted')
    if markup:
        ax[1].legend()
        ax[0].legend()
        ax[0].set_ylabel('z')
        if title:
            ax[0].set_title(title)
    if return_fig:
        return fig, ax
    else:
        return ax

fig, ax = plot_solution(zd, analytic)
fig.savefig(data_dir+'/analytic_solution.png', dpi=300)

b.change_scales(dealias)
q.change_scales(dealias)
b['g'] = analytic['b']['g']
q['g'] = analytic['q']['g']

dz = lambda A: de.Differentiate(A, dist.coords[-1])
lap = lambda A: dz(dz(A))

vars = [b, q]
nlbvp_taus = [τb1, τb2, τq1, τq2]

import dedalus.core.operators as op
import scipy.special as scp
def sym_diff(self, var):
    diff_map = {
            np.exp: lambda x: np.exp(x),
            np.tanh: lambda x: 0,
            scp.erf: lambda x: 0}
    arg = self.args[0]
    arg_diff = arg.sym_diff(var)
    return diff_map[self.func](arg) * arg_diff
op.UnaryGridFunction.sym_diff = sym_diff

# Stable nonlinear solution
problem = de.NLBVP(vars+nlbvp_taus, namespace=locals())
problem.add_equation('dt(b) - tau*P*lap(b) + lift(τb1, -1) + lift(τb2, -2) = γ*H(q-qs)*(q-qs)')
problem.add_equation('dt(q) - tau*S*lap(q) + lift(τq1, -1) + lift(τq2, -2) = - H(q-qs)*(q-qs)')
problem.add_equation('b(z=0) = 0')
problem.add_equation('b(z=Lz) = β + ΔT') # technically β*Lz
problem.add_equation('q(z=0) = q_surface') #*qs(z=0)')
problem.add_equation('q(z=Lz) = np.exp(α*ΔT)')

for system in ['subsystems']:
     logging.getLogger(system).setLevel(logging.WARNING)

start_time = time.time()
# for first tau loop, use analytic solution as first guess
need_guess = True

# on later tau loops, start with the highest k solution as the first guess, stored in sol
sol = {}

# reverse order on k solves
if args['--reverse_search_ks']:
    ks = ks[::-1]

for i, tau_i in enumerate(taus):
    for j, k_j in enumerate(ks):
        case_dir = 'tau_{:.2g}_k{:.2g}_nz{:d}'.format(tau_i, k_j, nz)
        if args['--erf']:
            case_dir += '_erf'
        if args['--Legendre']:
            case_dir += '_Legendre'

        if need_guess:
            logger.info('solving tau={:}, k={:}, starting from analytic'.format(tau_i, k_j))
            b.change_scales(dealias)
            q.change_scales(dealias)
            b['g'] = analytic['b']['g']
            q['g'] = analytic['q']['g']
            need_guess = False
        else:
            logger.info('solving tau={:}, k={:}, continuing from previous solution'.format(tau_i, k_j))

        tau['g'] = tau_i
        k['g'] = k_j
        solver = problem.build_solver()
        pert_norm = np.inf
        stop_iteration = int(args['--niter'])

        while pert_norm > tol and solver.iteration <= stop_iteration:
            solver.newton_iteration(damping=float(args['--damping']))
            pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
            logger.info("tau = {:.1g}, k = {:.0g}, L2 err = {:.1g}".format(tau['g'][0,0,0], k['g'][0,0,0], pert_norm))
            τb1_max = np.max(np.abs(τb1['g']))
            τb2_max = np.max(np.abs(τb2['g']))
            τq1_max = np.max(np.abs(τq1['g']))
            τq2_max = np.max(np.abs(τq2['g']))
            logger.debug("τ L2 errors: τb1={:.1g}, τb2={:.1g}, τq1={:.1g}, τq2={:.1g}".format(τb1_max,τb2_max,τq1_max,τq2_max))

        if solver.iteration > stop_iteration or not np.isfinite(pert_norm):
            logger.info("solution failed to converge")
            # logger.info("reverting to analytic solution as guess for next iteration")
            # need_guess = True
            logger.info(f"breaking k loop for tau = {tau_i}")
            break
        else:
            solution = solver.evaluator.add_file_handler(data_dir+'/'+case_dir+'/'+'drizzle_sol', mode='overwrite')
            solution.add_task(b)
            solution.add_task(q)
            solution.add_task(b + γ*q, name='m')
            solution.add_task(temp, name='T')
            solution.add_task(rh, name='rh')
            solution.add_task(tau, name='tau')
            solution.add_task(k, name='k')
            # these are basically asking for add_metadata
            solution.add_task(α_f, name='α')
            solution.add_task(β_f, name='β')
            solution.add_task(γ_f, name='γ')
        #    solution.add_metadata(niter, name='number of iterations')
            solver.evaluate_handlers()
            logger.info("wrote solution to {:}/{:}".format(data_dir, case_dir))

    need_guess = True
end_time = time.time()
logger.info("time to solve: {:.3g}s".format(end_time-start_time))
