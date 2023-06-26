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

    --use_analytic       Use an analytic solution as the beginning guess

    --Legendre

    --tau=<tau>          Tau parameter [default: 5e-5]
    --k=<k>              Tanh width of phase change [default: 1e3]

    --tolerance=<t>      Tolerance for convergence [default: 1e-6]

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

import os

from docopt import docopt
args = docopt(__doc__)

tau_in = float(args['--tau'])
k_in = float(args['--k'])

q_surface = float(args['--q0'])
nz = int(args['--nz'])

α = float(args['--alpha'])
β = float(args['--beta'])
γ = float(args['--gamma'])

ΔT = -1

Prandtl = 1
Prandtlm = 1
P = 1                                 #  diffusion on buoyancy
S = (Prandtlm/Prandtl)**(-1/2)        #  diffusion on moisture

data_dir = 'unsaturated_atm_alpha{:}_beta{:}_gamma{:}_q{:}'.format(args['--alpha'], args['--beta'], args['--gamma'], args['--q0'])
case_dir = 'tau_{:}_k{:}_nz{:d}'.format(args['--tau'], args['--k'], nz)

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

if not os.path.exists('{:s}/'.format(data_dir+'/'+case_dir)):
    os.mkdir('{:s}/'.format(data_dir+'/'+case_dir))


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
z = zb.local_grid(1)

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
k['g'] = k_in

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


#0.4832893544084419	-0.4588071140209613
zc_analytic = 0.4832893544084419
Tc_analytic = -0.4588071140209613

analytic = compute_analytic(zb.local_grid(dealias), zc_analytic, Tc_analytic)

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

fig, ax = plot_solution(zb.local_grid(dealias), analytic)
fig.savefig(data_dir+'/'+case_dir+'/analytic_solution.png', dpi=300)

if args['--use_analytic']:
    b.change_scales(dealias)
    q.change_scales(dealias)
    b['g'] = analytic['b']['g']
    q['g'] = analytic['q']['g']
else:
    # Stable linear solution as an intial guess
    problem = de.LBVP([b, q, τb1, τb2, τq1, τq2], namespace=locals())
    problem.add_equation('dt(b) - P*lap(b) + lift(τb1, -1) + lift(τb2, -2) = 0')
    problem.add_equation('dt(q) - S*lap(q) + lift(τq1, -1) + lift(τq2, -2) = 0')
    problem.add_equation('b(z=0) = 0')
    problem.add_equation('b(z=Lz) = β + ΔT') # technically β*Lz
    problem.add_equation('q(z=0) = q_surface')
    problem.add_equation('q(z=Lz) = np.exp(α*ΔT)')
    solver = problem.build_solver()
    solver.solve()


# Stable nonlinear solution
problem = de.NLBVP([b, q, τb1, τb2, τq1, τq2], namespace=locals())
problem.add_equation('dt(b) - tau*P*lap(b) + lift(τb1, -1) + lift(τb2, -2) = γ*H(q-qs)*(q-qs)')
problem.add_equation('dt(q) - tau*S*lap(q) + lift(τq1, -1) + lift(τq2, -2) = - H(q-qs)*(q-qs)')
problem.add_equation('b(z=0) = 0')
problem.add_equation('b(z=Lz) = β + ΔT') # technically β*Lz
problem.add_equation('q(z=0) = q_surface*qs(z=0)')
problem.add_equation('q(z=Lz) = np.exp(α*ΔT)')

for system in ['subsystems']:
     logging.getLogger(system).setLevel(logging.WARNING)

tau['g'] = tau_in
k['g'] = k_in
solver = problem.build_solver()
pert_norm = np.inf
while pert_norm > tol:
    solver.newton_iteration()
    pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
    logger.info("tau = {:.1g}, k = {:.0g}, L2 err = {:.1g}".format(tau['g'][0,0,0], k['g'][0,0,0], pert_norm))
    τb1_max = np.max(np.abs(τb1['g']))
    τb2_max = np.max(np.abs(τb2['g']))
    τq1_max = np.max(np.abs(τq1['g']))
    τq2_max = np.max(np.abs(τq2['g']))
    logger.debug("τ L2 errors: τb1={:.1g}, τb2={:.1g}, τq1={:.1g}, τq2={:.1g}".format(τb1_max,τb2_max,τq1_max,τq2_max))

solution = solver.evaluator.add_file_handler(data_dir+'/'+case_dir+'/'+'drizzle_sol', mode='overwrite')
solution.add_task(b)
solution.add_task(q)
solution.add_task(b + γ*q, name='m')
solution.add_task(temp, name='T')
solution.add_task(rh, name='rh')
solution.add_task(tau, name='tau')
solution.add_task(k, name='k')
solution.add_task(α_f, name='α')
solution.add_task(β_f, name='β')
solution.add_task(γ_f, name='γ')
solver.evaluate_handlers()
logger.info("wrote solution to {:}/{:}".format(data_dir, case_dir))


integ = lambda A: de.Integrate(A, 'z')
logger.info('values at boundaries')
logger.info('b {:.3g} to {:.3g}'.format(b(z=0).evaluate()['g'][0,0,0], b(z=Lz).evaluate()['g'][0,0,0]))
logger.info('T {:.3g} to {:.3g}'.format(temp(z=0).evaluate()['g'][0,0,0], temp(z=Lz).evaluate()['g'][0,0,0]))
logger.info('rh {:.3g} to {:.3g}'.format(rh(z=0).evaluate()['g'][0,0,0], rh(z=Lz).evaluate()['g'][0,0,0]))
if np.isclose(q_surface, 1):
    logger.info('L2(rh - 1) = {:.3g} '.format(np.sqrt(integ((rh-1)**2)).evaluate()['g'][0,0,0]))

def compute_L2_err(sol, analytic):
    return (integ(np.abs(sol-analytic))/integ(analytic)).evaluate()['g'][0,0,0]
L2_q = compute_L2_err(q, analytic['q'])
L2_b = compute_L2_err(b, analytic['b'])
L2_rh = compute_L2_err(rh, analytic['rh'])
L2_results = "L2:  q = {:.2g}, b={:.2g}, rh={:.2g}".format(L2_q, L2_b, L2_rh)
logger.info(L2_results)

m = (b + γ*q).evaluate()
sol = {'b':b, 'q':q, 'm':m, 'T':temp.evaluate(), 'rh':rh.evaluate(), 'γ':γ}
fig, ax = plot_solution(zb.local_grid(dealias), sol, title=L2_results)
plot_solution(zb.local_grid(dealias), analytic, ax=ax, linestyle='dashed')
fig.savefig(data_dir+'/'+case_dir+'/comparison_to_analytic_solution.png', dpi=300)
