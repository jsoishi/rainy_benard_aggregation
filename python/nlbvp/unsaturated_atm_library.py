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

    --nz=<nz>            Vertical (z) grid resolution [default: 256]
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)
for system in ['subsystems']:
     logging.getLogger(system).setLevel(logging.WARNING)
import matplotlib.pyplot as plt

import numpy as np
from dedalus import public as de

from fractions import Fraction

import os

from docopt import docopt
args = docopt(__doc__)

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

start_tau = 1e-3
stop_tau = 5e-5
taus = np.geomspace(start_tau, stop_tau, num=10)
ks = np.logspace(2, 3, num=5)

data_dir = 'unsaturated_atm_alpha{:}_gamma{:}_q{:}'.format(args['--alpha'], args['--gamma'], args['--q0'])

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

tol = 1e-3
IC = 'linear' #'LBVP' # 'LBVP' -> compute LBVP, 'linear' (or else) -> use linear ICs
verbose = True

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
lift1 = lambda A, n: de.Lift(A, zb1, n)
lift = lambda A, n: de.Lift(A, zb2, n)

ex, ey, ez = coords.unit_vector_fields(dist)

z_grid = dist.Field(name='z_grid', bases=zb)
z_grid['g'] = z

temp = b - β*z_grid
temp.name = 'T'

qs = np.exp(α*temp)
rh = q*np.exp(-α*temp)

tau = dist.Field(name='tau')
k = dist.Field(name='k')
H = lambda A: 0.5*(1+np.tanh(k*A))

logger.info('solving LBVP for initial guess; tau, k not needed.')
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

dt = lambda A: 0*A

# Stable nonlinear solution
problem = de.NLBVP([b, q, τb1, τb2, τq1, τq2], namespace=locals())
problem.add_equation('dt(b) - P*lap(b) + lift(τb1, -1) + lift(τb2, -2) = γ*H(q-qs)*(q-qs)/tau')
problem.add_equation('dt(q) - S*lap(q) + lift(τq1, -1) + lift(τq2, -2) = - H(q-qs)*(q-qs)/tau')
problem.add_equation('b(z=0) = 0')
problem.add_equation('b(z=Lz) = β + ΔT') # technically β*Lz
problem.add_equation('q(z=0) = q_surface*qs(z=0)')
problem.add_equation('q(z=Lz) = np.exp(α*ΔT)')

# for first tau loop, use LBVP or linear solution as first guess
need_guess = False

# on later tau loops, start with the highest k solution as the first guess, stored in sol
sol = {}

for i, tau_i in enumerate(taus):
    if i == 0:
        # on the first sweep, start at low tau and increase tau
        k_set = ks
    else:
        # reverse order on k solves after the first sweep
        k_set = ks[::-1]

    for j, k_j in enumerate(k_set):
        case_dir = 'tau_{:.2g}_k{:.2g}_nz{:d}'.format(tau_i, k_j, nz)

        # confused why we need this mkdir call, given filehandler, but let's go with it
        if not os.path.exists('{:s}/'.format(data_dir+'/'+case_dir)):
            os.mkdir('{:s}/'.format(data_dir+'/'+case_dir))

        if need_guess:
            logger.info('solving tau={:}, k={:}, starting from guess'.format(tau_i, k_j))
            b.change_scales(1)
            q.change_scales(1)
            sol['b'].change_scales(1)
            sol['q'].change_scales(1)
            b['g'] = sol['b']['g']
            q['g'] = sol['q']['g']
            need_guess = False
        else:
            logger.info('solving tau={:}, k={:}, continuing from previous solution'.format(tau_i, k_j))

        tau['g'] = tau_i
        k['g'] = k_j

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
        # work around for file handlers:
        solver.evaluator.evaluate_handlers([solution], timestep=0, wall_time=0, sim_time=0, iteration=0, world_time=0)
        #solution.process()
        logger.info("wrote solution to {:}/{:}".format(data_dir, case_dir))
        if j == 0:
            need_guess = False
            if i != 0:
                logger.info('saving first solution as first guess for next loop')
                sol['b'] = b.copy()
                sol['q'] = q.copy()

    if i == 0:
        logger.info('saving last solution as first guess for next loop')
        sol['b'] = b.copy()
        sol['q'] = q.copy()

    need_guess = True
