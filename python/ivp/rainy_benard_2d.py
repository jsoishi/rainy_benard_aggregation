"""
Dedalus script for calculating dynamic solutions to the Rainy-Benard system.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves IVPs for an existing atmospheres, solved for by scripts in the nlbvp section.

Usage:
    rainy_benard.py <case> [options]

Options:
    <case>            Case to build IVP around

    --Rayleigh=<Ra>   Rayleigh number to test [default: 1e5]
    --tau=<tau>       Tau to solve; if not set, use tau of background
    --aspect=<a>      Aspect ratio of domain, Lx/Lz [default: 10]

    --nondim=<n>      Non-Nondimensionalization [default: buoyancy]

    --top-stress-free     Stress-free upper boundary
    --stress-free         Stress-free both boundaries

    --nz=<nz>         Number z coeffs to use in IVP; if not set, uses resolution of background solution
    --nx=<nx>         Number of x coeffs to use in IVP; if not set, scales nz by aspect

    --max_dt=<dt>     Largest timestep to use; should be set by oscillation timescales of waves (Brunt) [default: 1]

    --verbose         Show plots on screen
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)

import numpy as np
import dedalus.public as de
from dedalus.extras import flow_tools
import h5py

from docopt import docopt
args = docopt(__doc__)

case = args['<case>']
with h5py.File(case+'/drizzle_sol/drizzle_sol_s1.h5', 'r') as f:
    sol = {}
    for task in f['tasks']:
        sol[task] = f['tasks'][task][0,0,0][:]
    sol['z'] = f['tasks']['b'].dims[3][0][:]
    tau_in = sol['tau'][0]
    k = sol['k'][0]
    α = sol['α'][0]
    β = sol['β'][0]
    γ = sol['γ'][0]

aspect = float(args['--aspect'])

nz_sol = sol['z'].shape[0]
if args['--nz']:
    nz = int(float(args['--nz']))
else:
    nz = nz_sol
if args['--nx']:
    nx = int(float(args['--nx']))
else:
    nx = int(aspect)*nz

if args['--tau']:
    tau = float(args['--tau'])
else:
    tau = tau_in

case_dir = 'rainy_benard_Ra{:}_tau{:.2g}_k{:.2g}_nz{:d}_nx{:d}'.format(args['--Rayleigh'], tau, k, nz, nx)

dealias = 3/2
dtype = np.float64

Prandtlm = 1
Prandtl = 1
Rayleigh = float(args['--Rayleigh'])

Lz = 1
Lx = aspect

coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)
xb = de.RealFourier(coords.coords[0], size=nx, bounds=(0, Lx), dealias=dealias)
zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
x = xb.local_grid(1)
z = zb.local_grid(1)

b0 = dist.Field(name='b0', bases=zb)
q0 = dist.Field(name='q0', bases=zb)

zb_sol = de.ChebyshevT(coords.coords[2], size=nz_sol, bounds=(0, Lz), dealias=dealias)
b0_sol = dist.Field(name='b0_sol', bases=zb_sol)
q0_sol = dist.Field(name='q0_sol', bases=zb_sol)

b0_sol['g'] = sol['b']
q0_sol['g'] = sol['q']

scale_ratio = nz/nz_sol
b0_sol.change_scales(scale_ratio)
q0_sol.change_scales(scale_ratio)

logger.info('rescaling background from {:} to {:} coeffs (ratio: {:})'.format(nz_sol, nz, scale_ratio))
b0['g'] = b0_sol['g']
q0['g'] = q0_sol['g']

bases = (xb, zb)

p = dist.Field(name='p', bases=bases)
u = dist.VectorField(coords, name='u', bases=bases)
b = dist.Field(name='b', bases=bases)
q = dist.Field(name='q', bases=bases)

τp = dist.Field(name='τp')
τu1 = dist.VectorField(coords, name='τu1', bases=xb)
τu2 = dist.VectorField(coords, name='τu2', bases=xb)
τb1 = dist.Field(name='τb1', bases=xb)
τb2 = dist.Field(name='τb2', bases=xb)
τq1 = dist.Field(name='τq1', bases=xb)
τq2 = dist.Field(name='τq2', bases=xb)

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

T = b
qs = q0*np.exp(α*T)

vars = [p, u, b, q, τp, τu1, τu2, τb1, τb2, τq1, τq2]
problem = de.IVP(vars, namespace=locals())

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

sech = lambda A: 1/np.cosh(A)
scrN = (H(q0 - qs0) + (q0 - qs0)*k/2*sech(k*(q0 - qs0))**2).evaluate()
scrN.name='scrN'
scrN_g = de.Grid(scrN).evaluate()

H_q0 = ((q0 - qs0)*H(q0 - qs0)).evaluate()
H_q0.name='((q0-qs0)*H(q0-qs0))'
H_q0_g = de.Grid(H_q0).evaluate()

qs = np.exp(α*T)

problem.add_equation('div(u) + τp + 1/PdR*dot(lift(τu2,-1),ez) = 0')
problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b*ez + lift(τu1, -1) + lift(τu2, -2) = -(u@grad(u))')
# problem.add_equation('dt(b) - P*lap(b) + u@grad(b0) - γ/tau*(q-α*qs0*b)*scrN + lift(τb1, -1) + lift(τb2, -2) = - (u@grad(b)) + γ/tau*((q-qs)*H(q-qs) - (q-α*qs0*b)*scrN_g)')
# problem.add_equation('dt(q) - S*lap(q) + u@grad(q0) + 1/tau*(q-α*qs0*b)*scrN + lift(τq1, -1) + lift(τq2, -2) = - (u@grad(q)) - 1/tau*((q-qs)*H(q-qs) - (q-α*qs0*b)*scrN_g)')
problem.add_equation('dt(b) - P*lap(b) + u@grad(b0) + lift(τb1, -1) + lift(τb2, -2) = - (u@grad(b)) + γ/tau*((q+q0-qs-qs0)*H(q+q0-qs-qs0)-H_q0_g)')
problem.add_equation('dt(q) - S*lap(q) + u@grad(q0) + lift(τq1, -1) + lift(τq2, -2) = - (u@grad(q)) - 1/tau*((q+q0-qs-qs0)*H(q+q0-qs-qs0)-H_q0_g)')
problem.add_equation('b(z=0) = 0')
problem.add_equation('b(z=Lz) = 0')
problem.add_equation('q(z=0) = 0')
problem.add_equation('q(z=Lz) = 0')
if args['--stress-free']:
    problem.add_equation('ez@u(z=0) = 0')
    problem.add_equation('ez@(ex@e(z=0)) = 0')
    problem.add_equation('ez@(ey@e(z=0)) = 0')
else:
    problem.add_equation('u(z=0) = 0')
if args['--top-stress-free'] or args['--stress-free']:
    problem.add_equation('ez@u(z=Lz) = 0')
    problem.add_equation('ez@(ex@e(z=Lz)) = 0')
    problem.add_equation('ez@(ey@e(z=Lz)) = 0')
else:
    problem.add_equation('u(z=Lz) = 0')
problem.add_equation('integ(p) = 0')

# initial conditions
amp = 1e-4

noise = dist.Field(name='noise', bases=bases)
noise.fill_random('g', seed=42, distribution='normal', scale=amp) # Random noise
noise.low_pass_filter(scales=0.25)

# noise ICs in buoyancy
b['g'] = noise['g']*np.cos(np.pi/2*z/Lz)

ts = de.SBDF2
cfl_safety_factor = 0.2
solver = problem.build_solver(ts)
solver.stop_iteration = run_time_iter

Δt = max_Δt = float(args['--max_dt'])
cfl = flow_tools.CFL(solver, Δt, safety=cfl_safety_factor, cadence=1, threshold=0.1,
                     max_change=1.5, min_change=0.5, max_dt=max_Δt)
cfl.add_velocity(u)

report_cadence = 10

integ = lambda A: de.Integrate(de.Integrate(A, 'x'), 'z')
avg = lambda A: integ(A)/(Lx*Lz)
x_avg = lambda A: de.Integrate(A, 'x')/(Lx)

Re = np.sqrt(u@u)/PdR
KE = 0.5*np.sqrt(u@u)

flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(Re, name='Re')
flow.add_property(KE, name='KE')
flow.add_property(np.sqrt(τu1@τu1), name='|τu1|')
flow.add_property(np.sqrt(τu2@τu2), name='|τu2|')
flow.add_property(np.abs(τb1), name='|τb1|')
flow.add_property(np.abs(τb2), name='|τb2|')
flow.add_property(np.abs(τq1), name='|τq1|')
flow.add_property(np.abs(τq2), name='|τq2|')
flow.add_property(np.abs(τp), name='|τp|')

vol = Lx*Lz

good_solution = True
KE_avg = 0
while solver.proceed and good_solution:
    # advance
    solver.step(Δt)
    if solver.iteration % report_cadence == 0:
        τ_max = np.max([flow.max('|τu1|'),flow.max('|τu2|'),flow.max('|τb1|'),flow.max('|τb2|'),flow.max('|τq1|'),flow.max('|τq2|'),flow.max('|τp|')])
        Re_max = flow.max('Re')
        Re_avg = flow.volume_integral('Re')/vol
        KE_avg = flow.volume_integral('KE')/vol
        log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:5.1e}'.format(solver.iteration, solver.sim_time, Δt)
        log_string += ', KE: {:.2g}, Re: {:.2g} ({:.2g})'.format(KE_avg, Re_avg, Re_max)
        log_string += ', τ: {:.2g}'.format(τ_max)
        logger.info(log_string)
    Δt = cfl.compute_timestep()
    good_solution = np.isfinite(Δt)*np.isfinite(KE_avg)
