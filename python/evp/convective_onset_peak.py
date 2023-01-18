"""
Dedalus script for determining instability of static drizzle solutions to the Rainy-Benard system of equations.  This script searches for a peak growth rate in continuous kx using scipy optimization routines, and then uses a bracketing search to determine the critical Rayleigh number.

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

    --Rayleigh=<Ra>   Initial Rayleigh to begin search at; appears better to start from above onset [default: 1e5]
    --kx=<kx>         Initial kx to begin search at [default: 5]

    --nondim=<n>      Non-Nondimensionalization [default: buoyancy]

    --target=<targ>   Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>     Target number of eigenvalues to search for [default: 20]

    --method=<m>      Method for Ra search; valid entries are 'line' or '2d' [default: line]

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
    tau_in = sol['tau'][0]
    k = sol['k'][0]
    α = sol['α'][0]
    β = sol['β'][0]
    γ = sol['γ'][0]

nz = sol['z'].shape[0]

dealias = 3/2
dtype = np.complex128

Prandtlm = 1
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
Rayleigh = dist.Field(name='Ra_c')

# follows Roberts 1972 convention, eq 1.1, 2.8
dx = lambda A: 1j*kx*A
dy = lambda A: 0*A #1j*kx*A # try 2-d mode onset
dz = lambda A: de.Differentiate(A, coords['z'])

grad = lambda A: de.Gradient(A, coords)
div = lambda A:  dx(A@ex) + dy(A@ey) + dz(A@ez)
grad = lambda A: dx(A)*ex + dy(A)*ey + dz(A)*ez
lap = lambda A: dx(dx(A)) + dy(dy(A)) + dz(dz(A))

vars = [p, u, b, q, τp, τu1, τu2, τb1, τb2, τq1, τq2]
# fix Ra, find omega
dt = lambda A: ω*A
ω = dist.Field(name='ω')
problem = de.EVP(vars, eigenvalue=ω, namespace=locals())
#Ras = np.logspace(4,5,num=5)

nondim = args['--nondim']
if nondim == 'diffusion':
    P = 1                      #  diffusion on buoyancy. Always = 1 in this scaling.
    S = Prandtlm               #  diffusion on moisture  k_q / k_b
    PdR = Prandtl              #  diffusion on momentum
    PtR = Prandtl*Rayleigh     #  Prandtl times Rayleigh = buoyancy force
elif nondim == 'buoyancy':
    P = (Rayleigh * Prandtl)**(-1/2)         #  diffusion on buoyancy
    S = (Rayleigh * Prandtlm)**(-1/2)        #  diffusion on moisture
    PdR = (Prandtl / Rayleigh)**(1/2)        #  diffusion on momentum
    PtR = 1
    #tau_in /=                     # think through what this should be
else:
    raise ValueError('nondim {:} not in valid set [diffusion, buoyancy]'.format(nondim))

tau['g'] = tau_in

sech = lambda A: 1/np.cosh(A)
scrN = (H(q0 - qs0) + (q0 - qs0)*k/2*sech(k*(q0 - qs0))**2).evaluate()
scrN.name='scrN'

problem.add_equation('div(u) + τp + 1/PdR*dot(lift(τu2,-1),ez) = 0')
problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b*ez + lift(τu1, -1) + lift(τu2, -2) = 0')
problem.add_equation('dt(b) - P*lap(b) + dot(u, grad(b0)) - γ/tau*(q-α*qs0*b)*scrN + lift(τb1, -1) + lift(τb2, -2) = 0')
problem.add_equation('dt(q) - S*lap(q) + dot(u, grad(q0)) + 1/tau*(q-α*qs0*b)*scrN + lift(τq1, -1) + lift(τq2, -2) = 0')
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

# fix Ra, find omega
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

def critical_finder_2d(kx_Ra):
    rate = compute_growth_rate(kx_Ra[0], np.exp(kx_Ra[1]))
    # looking for nearest zero values, so return abs(Re(σ))
    return np.abs(rate.real)

def peak_growth_rate(*args):
    rate = compute_growth_rate(*args)
    # flip sign so minimize finds maximum
    return -1*rate.real

def critical_kx(Ra_in, log_search):
    if log_search:
        Ra_i = np.exp(Ra_in)
        tol = 1e-3
    else:
        Ra_i = np.exp(Ra_in)
        tol = 1e-5
    kx_i = kx_dict['peak_kx']
    bounds = sciop.Bounds(lb=1, ub=10)
    result = sciop.minimize(peak_growth_rate, kx_i, args=(Ra_i), bounds=bounds, method='Nelder-Mead', tol=tol)
    σ = compute_growth_rate(result.x[0], Ra_i)
    logger.info('ω = {:} at kx = {:}, Ra = {:}, success = {:}'.format(σ, result.x[0], Ra_i, result.success))
    # update outer variable for next loop
    kx_dict['peak_kx'] = np.abs(result.x[0])
    return np.abs(σ.real)

import scipy.optimize as sciop
kx_start = float(args['--kx'])
Rayleigh_start = float(args['--Rayleigh'])
if method == '2d':
    result = sciop.minimize(critical_finder, x0=[kx_start, np.log(Rayleigh_start)])
    σ = compute_growth_rate(result.x[0], np.exp(result.x[1]))
elif method == 'line':
    kx_dict = {'peak_kx':kx_start}
    logger.info('beginning linear search about Ra={:}, kx={:}'.format(Rayleigh_start, kx_dict['peak_kx']))
    result = sciop.minimize(critical_kx, np.log(Rayleigh_start), args=(False))
    # logger.info('beginning linear search about Ra={:}, kx={:}'.format(result.x[0], kx_dict['peak_kx']))
    # result = sciop.minimize(critical_kx, result.x[0], args=(False))
else:
    raise ValueError("search method {:} not in valid set ['line', '2d']".format(method))

logger.info('ω = {:} at kx = {:}, Ra = {:}, success = {:}'.format(σ, result.x[0], np.exp(result.x[1]), result.success))
print(result)
