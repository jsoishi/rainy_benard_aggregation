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

    --tau=<tau>          Tau parameter [default: 5e-5]
    --k=<k>              Tanh width of phase change [default: 1e3]

    --tolerance=<t>      Tolerance for convergence [default: 1e-3]

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

data_dir = 'unsaturated_atm_alpha{:}_beta_{:}_gamma{:}_q{:}'.format(args['--alpha'], args['--beta'], args['--gamma'], args['--q0'])
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
zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
z = zb.local_grid(1)

b = dist.Field(name='b', bases=zb)
q = dist.Field(name='q', bases=zb)

τb1 = dist.Field(name='τb1')
τb2 = dist.Field(name='τb2')
τq1 = dist.Field(name='τq1')
τq2 = dist.Field(name='τq2')

zb1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
zb2 = zb.clone_with(a=zb.a+2, b=zb.b+2)
#lift1 = lambda A, n: de.Lift(A, zb1, n)
lift = lambda A, n: de.Lift(A, zb, n)

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

dt = lambda A: 0*A
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

α_f = dist.Field()
α_f['g'] = α
β_f = dist.Field()
β_f['g'] = β
γ_f = dist.Field()
γ_f['g'] = γ

dt = lambda A: 0*A

# Stable nonlinear solution
problem = de.NLBVP([b, q, τb1, τb2, τq1, τq2], namespace=locals())
problem.add_equation('dt(b) - P*lap(b) + lift(τb1, -1) + lift(τb2, -2) = γ*H(q-qs)*(q-qs)/tau')
problem.add_equation('dt(q) - S*lap(q) + lift(τq1, -1) + lift(τq2, -2) = - H(q-qs)*(q-qs)/tau')
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
# work around for file handlers:
solver.evaluator.evaluate_handlers([solution], timestep=0, wall_time=0, sim_time=0, iteration=0, world_time=0)
#solution.process()
logger.info("wrote solution to {:}/{:}".format(data_dir, case_dir))


integ = lambda A: de.Integrate(A, 'z')
logger.info('values at boundaries')
logger.info('b {:.3g} to {:.3g}'.format(b(z=0).evaluate()['g'][0,0,0], b(z=Lz).evaluate()['g'][0,0,0]))
logger.info('T {:.3g} to {:.3g}'.format(temp(z=0).evaluate()['g'][0,0,0], temp(z=Lz).evaluate()['g'][0,0,0]))
logger.info('rh {:.3g} to {:.3g}'.format(rh(z=0).evaluate()['g'][0,0,0], rh(z=Lz).evaluate()['g'][0,0,0]))
if np.isclose(q_surface, 1):
    logger.info('L2(rh - 1) = {:.3g} '.format(np.sqrt(integ((rh-1)**2)).evaluate()['g'][0,0,0]))
