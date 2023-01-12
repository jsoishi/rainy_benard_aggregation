"""
Dedalus script for determining instability of static drizzle solutions to the Rainy-Benard system of equations.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves EVPs for an existing atmospheres, solved for by scripts in the nlbvp section.

Roberts, G.O., 1972,
``Dynamo action of fluid motions with two-dimensional periodicity''

Usage:
    convective_onset.py <cases>... [options]

Options:
    <cases>           Case (or cases) to plot results from

    --method=<m>      Method of onset searching [default: Rayleigh]
    --Rayleigh=<Ra>   Rayleigh number to test [default: 1e5]

    --target=<targ>   Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>     Target number of eigenvalues to search for [default: 20]

    --dense           Solve densely for all eigenvalues (slow)

    --verbose         Show plots on screen
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

method = args['--method']
N_evals = int(float(args['--eigs']))
target = float(args['--target'])

for case in args['<cases>']:
    f = h5py.File(case+'/drizzle_sol/drizzle_sol_s1.h5', 'r')
    sol = {}
    for task in f['tasks']:
        sol[task] = f['tasks'][task][0,0,0][:]
    sol['z'] = f['tasks']['b'].dims[3][0][:]
    tau = sol['tau'][0]
    k = sol['k'][0]
    α = sol['α'][0]
    β = sol['β'][0]
    γ = sol['γ'][0]

nz = sol['z'].shape[0]

dealias = 3/2
dtype = np.complex128

Prandtl = 1
Rayleigh = float(args['--Rayleigh'])


Lz = 1
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)
dealias = 2
zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
z = zb.local_grid(1)

b0 = dist.Field(name='b0', bases=zb)
q0 = dist.Field(name='q0', bases=zb)
b0['g'] = sol['b']
q0['g'] = sol['q']

p = dist.Field(name='p', bases=zb)
u = dist.VectorField(coords, name='u', bases=zb)
b = dist.Field(name='b', bases=zb)
q = dist.Field(name='q', bases=zb)

τp = dist.Field(name='τp')
τu1 = dist.VectorField(coords, name='τu1')
τu2 = dist.VectorField(coords, name='τu2')
τb1 = dist.Field(name='τb1')
τb2 = dist.Field(name='τb2')
τq1 = dist.Field(name='τq1')
τq2 = dist.Field(name='τq2')

zb1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
zb2 = zb.clone_with(a=zb.a+2, b=zb.b+2)
lift1 = lambda A, n: de.Lift(A, zb1, n)
lift = lambda A, n: de.Lift(A, zb2, n)

ex, ey, ez = coords.unit_vector_fields(dist)

H = lambda A: 0.5*(1+np.tanh(k*A))

z_grid = dist.Field(name='z_grid', bases=zb)
z_grid['g'] = z

T0 = b0 - β*z_grid
qs0 = np.exp(α*T0)

tau = dist.Field(name='tau')
kx = dist.Field(name='kx')
# follows Roberts 1972 convention, eq 1.1, 2.8
dx = lambda A: 1j*kx*A
dy = lambda A: 1j*kx*A
dz = lambda A: de.Differentiate(A, coords['z'])

grad = lambda A: de.Gradient(A, coords) #+ 1j*kx*A*ex
div = lambda A:  dx(A@ex) + dy(A@ey) + dz(A@ez)
grad = lambda A: dx(A)*ex + dy(A)*ey + dz(A)*ez
lap = lambda A: dx(dx(A)) + dy(dy(A)) + dz(dz(A))
# curl = lambda A: (dy(A@ez)-dz(A@ey))*ex + (dz(A@ex)-dx(A@ez))*ey + (dx(A@ey)-dy(A@ex))*ez

vars = [p, u, b, q, τp, τu1, τu2, τb1, τb2, τq1, τq2]
if method == 'omega':
    # fix Ra, find omega
    dt = lambda A: ω*A
    ω = dist.Field(name='ω')
    problem = de.EVP(vars, eigenvalue=ω, namespace=locals())
elif method == 'Rayleigh':
    # direct onset, find Ra
    dt = lambda A: 0*A
    Rayleigh = dist.Field(name='Ra_c')
    problem = de.EVP(vars, eigenvalue=Rayleigh, namespace=locals())
else:
    raise ValueError('method = {:} is not in [omega, Rayleigh]'.format(method))

P = 1                      #  diffusion on buoyancy. Always = 1 in this scaling.
S = 1                      #  diffusion on moisture  k_q / k_b
PdR = Prandtl              #  diffusion on momentum
PtR = Prandtl*Rayleigh     #  Prandtl times Rayleigh = buoyancy force

problem.add_equation('div(u) + τp + 1/PdR*dot(lift(τu2,-1),ez) = 0')
problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b*ez + lift(τu1, -1) + lift(τu2, -2) = 0')
problem.add_equation('dt(b) - P*lap(b) + dot(u, grad(b0)) - γ*H(q0-qs0)*(q)/tau + lift(τb1, -1) + lift(τb2, -2) = 0')
problem.add_equation('dt(q) - S*lap(q) + dot(u, grad(q0)) + H(q0-qs0)*(q)/tau + lift(τq1, -1) + lift(τq2, -2) = 0')
problem.add_equation('b(z=0) = 0')
problem.add_equation('b(z=Lz) = 0')
problem.add_equation('q(z=0) = 0')
problem.add_equation('q(z=Lz) = 0')
problem.add_equation('u(z=0) = 0')
problem.add_equation('u(z=Lz) = 0')
problem.add_equation('integ(p) = 0')
solver = problem.build_solver()

dlog = logging.getLogger('subsystems')
dlog.setLevel(logging.WARNING)

def peak_growth(kx_i):
    kx['g'] = kx_i
    if args['--dense']:
        solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
    else:
        solver.solve_sparse(solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
    i_evals = np.argsort(solver.eigenvalues.real)
    evals = solver.eigenvalues[i_evals]

    sort_indices = np.argsort(np.abs(solver.eigenvalues.real))
    eigenvalues = solver.eigenvalues.real[sort_indices]
    eigenvalues = eigenvalues[eigenvalues > 0]
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    evals = eigenvalues
    logger.info('kx = {:}, ω = {:}'.format(kx['g'], evals)) #evals[-1]))
    # turn max into min, for minimize
    return -1*evals[-1].real




# import scipy.optimize as sciop
# kx_start = 0.5
# logger.info('starting optimization solve')
# result = sciop.minimize(peak_growth, kx_start)
# logger.info('solve complete')
# if result.success:
#     logger.info('fastest growing mode, ω = {:} at kx = {:}'.format(-1*result.fun, result.x[0]))
# else:
#     logger.info('solver failed to converge: {}'.format(result.message))


# fix Ra, find omega
Rayleigh = dist.Field(name='Ra_c')
def compute_growth_rate(kx_i, Ra_i):
    kx['g'] = kx_i
    Rayleigh['g'] = Ra_i

    if args['--dense']:
        solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
    else:
        solver.solve_sparse(solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
    i_evals = np.argsort(solver.eigenvalues.real)
    evals = solver.eigenvalues[i_evals]
    return evals[-1]


growth_rates = {}

Ras = np.logspace(4,6,num=3)
kxs = np.logspace(-1, 2, num=30)
print(Ras)
for Ra_i in Ras:
    σ = []
    for kx_i in kxs:
        σ_i = compute_growth_rate(kx_i, Ra_i)
        σ.append(σ_i)
        logger.info('Ra = {:.1g}, kx = {:.2g}, σ = {:.2g}'.format(Ra_i, kx_i, σ_i))
    growth_rates[Ra_i] = np.array(σ)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
peak_σ = -np.inf
for Ra in growth_rates:
    σ = growth_rates[Ra]
    peak_σ = max(peak_σ, np.max(σ))
    ax.plot(kxs, σ, label='Ra = {:.2g}'.format(Ra))
ax.set_xscale('log')
ax.set_ylim(-0.2, 0.05)
ax.legend()
ax.axhline(y=0, linestyle='dashed', color='xkcd:grey', alpha=0.5)
ax.set_xlim(3e-1,1e1)
ax.set_title(r'$\gamma$ = {:}, $\beta$ = {:}, $\tau$ = {:}'.format(γ,β, tau))
ax.set_ylabel('growth rate')
ax.set_xlabel('$k_x$')
fig.savefig(case+'/growth_curves.png', dpi=300)
