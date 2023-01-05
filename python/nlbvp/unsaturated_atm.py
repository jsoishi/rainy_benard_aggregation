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

    --Nz=<Nz>            Vertical (z) grid resolution [default: 256]
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
nz = int(args['--Nz'])

α = float(args['--alpha'])
β = float(args['--beta'])
γ = float(args['--gamma'])

ΔT = -1

Prandtl = 1
Prandtlm = 1
P = 1                                 #  diffusion on buoyancy
S = (Prandtlm/Prandtl)**(-1/2)        #  diffusion on moisture

data_dir = 'unsaturated_atm_alpha{:}_gamma{:}_q{:}'.format(args['--alpha'], args['--gamma'], args['--q0'])
case_dir = 'tau_{:}_k{:}_nz{:d}'.format(args['--tau'], args['--k'], nz)

if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))
if not os.path.exists('{:s}/'.format(data_dir+'/'+case_dir)):
    os.mkdir('{:s}/'.format(data_dir+'/'+case_dir))

tol = 1e-3
IC = 'linear' #'LBVP' # 'LBVP' -> compute LBVP, 'linear' (or else) -> use linear ICs
verbose = True

Lz = 1

# start_tau = 1e-3
# stop_tau = 5e-5
# taus = np.logspace(np.log10(start_tau), np.log10(stop_tau), num=10)
# ks = np.logspace(2, 3, num=4)

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


def plot_solution(solution, title=None, mask=None, linestyle=None, ax=None):
    b = solution['b']
    q = solution['q']
    m = solution['m']
    T = solution['T']
    rh = solution['rh']

    for f in [b, q, m, T, rh]:
        f.change_scales(1)

    if mask is None:
        mask = np.ones_like(z, dtype=bool)
    if ax is None:
        fig, ax = plt.subplots(ncols=2)
        markup = True
    else:
        for axi in ax:
            axi.set_prop_cycle(None)
        markup = False
    ax[0].plot(b['g'][mask],z[mask], label='$b$', linestyle=linestyle)
    ax[0].plot(γ*q['g'][mask],z[mask], label='$\gamma q$', linestyle=linestyle)
    ax[0].plot(m['g'][mask],z[mask], label='$b+\gamma q$', linestyle=linestyle)

    ax[1].plot(T['g'][mask],z[mask], label='$T$', linestyle=linestyle)
    ax[1].plot(q['g'][mask],z[mask], label='$q$', linestyle=linestyle)
    ax[1].plot(rh['g'][mask],z[mask], label='$r_h$', linestyle=linestyle)

    if markup:
        ax[1].legend()
        ax[0].legend()
        ax[0].set_ylabel('z')
        if title:
            ax[0].set_title(title)
    return ax


from scipy.optimize import newton
from scipy.interpolate import interp1d

def find_zc(sol, ε=1e-3, root_finding = 'inverse'):
    rh = sol['rh']
    rh.change_scales(1)
    f = interp1d(z[0,0,:], rh['g'][0,0,:])
    if root_finding == 'inverse':
        # invert the relationship and use interpolation to find where r_h = 1-ε (approach from below)
        f_i = interp1d(rh['g'][0,0,:], z[0,0,:]) #inverse
        zc = f_i(1-ε)
    elif root_finding == 'discrete':
        # crude initial emperical zc; look for where rh-1 ~ 0, in lower half of domain.
        zc = z[0,0,np.argmin(np.abs(rh['g'][0,0,0:int(nz/2)]-1))]
#    if zc is None:
#        zc = 0.2
#    zc = newton(f, 0.2)
    return zc


if IC == 'LBVP':
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
else:
    b['g'] = (β + ΔT)*z
    q['g'] = (1-z+np.exp(α*ΔT))

print('b: {:.2g} -- {:.2g}'.format(b(z=0).evaluate()['g'][0,0,0], b(z=Lz).evaluate()['g'][0,0,0]))
print('q: {:.2g} -- {:.2g}'.format(q(z=0).evaluate()['g'][0,0,0], q(z=Lz).evaluate()['g'][0,0,0]))

LBVP_sol = {'b':b.copy(), 'q':q.copy(), 'm':(b+γ*q).evaluate().copy(), 'T':temp.evaluate().copy(), 'rh':rh.evaluate().copy()}
if verbose:
    plot_solution(LBVP_sol, title='LBVP solution')
if IC == 'LBVP':
    zc = find_zc(LBVP_sol)
    print('LBVP zc = {:.3}'.format(zc))
    LBVP_sol['zc'] = zc


# In[10]:


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
NLBVP_sol = {'b':b.copy(), 'q':q.copy(),
             'm':(b+γ*q).evaluate().copy(), 'T':temp.evaluate().copy(), 'rh':rh.evaluate().copy(),
             'tau':tau['g'][0,0,0], 'k':k['g'][0,0,0]}
zc = find_zc(NLBVP_sol)
NLBVP_sol['zc'] = zc
logger.info('tau = {:.1g}, k = {:.0g}, zc = {:.2g}'.format(tau['g'][0,0,0], k['g'][0,0,0], zc))

value = rh.evaluate()
value.change_scales(1)
mask = (value['g'] >= 1-0.01)
ax = plot_solution(NLBVP_sol, title='NLBVP solution', mask=mask, linestyle='solid')
mask = (value['g'] < 1-0.01)
plot_solution(NLBVP_sol, title='NLBVP solution', mask=mask, linestyle='dashed', ax=ax)
print('zc = {:.3g}'.format(NLBVP_sol['zc']))
print('zc = {:.3g}'.format(find_zc(NLBVP_sol)))


plt.show()
