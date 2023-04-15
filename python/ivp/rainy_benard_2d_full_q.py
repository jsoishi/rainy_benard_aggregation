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

    --Rayleigh=<Ra>   Rayleigh number [default: 1e5]
    --tau=<tau>       Tau to solve; if not set, use tau of background
    --aspect=<a>      Aspect ratio of domain, Lx/Lz [default: 10]

    --nondim=<n>      Non-Nondimensionalization [default: buoyancy]

    --top-stress-free     Stress-free upper boundary
    --stress-free         Stress-free both boundaries

    --nz=<nz>         Number z coeffs to use in IVP; if not set, uses resolution of background solution
    --nx=<nx>         Number of x coeffs to use in IVP; if not set, scales nz by aspect

    --max_dt=<dt>     Largest timestep to use; should be set by oscillation timescales of waves (Brunt) [default: 1]

    --run_time_diff=<rtd>      Run time, in diffusion times [default: 1]
    --run_time_buoy=<rtb>      Run time, in buoyancy times
    --run_time_iter=<rti>      Run time, number of iterations; if not set, n_iter=np.inf

    --label=<label>   Label to add to output directory
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)

import numpy as np
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

data_dir = case+'/rainy_benard_Ra{:}_tau{:.2g}_k{:.2g}_nz{:d}_nx{:d}'.format(args['--Rayleigh'], tau, k, nz, nx)

if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

import dedalus.public as de
from dedalus.extras import flow_tools

logger.info('α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(α,β,γ,tau_in, k))

Prandtlm = 1
Prandtl = 1
Rayleigh = float(args['--Rayleigh'])

run_time_buoy = args['--run_time_buoy']
if run_time_buoy != None:
    run_time_buoy = float(run_time_buoy)
else:
    run_time_buoy = float(args['--run_time_diff'])*np.sqrt(Rayleigh)

run_time_iter = args['--run_time_iter']
if run_time_iter != None:
    run_time_iter = int(float(run_time_iter))
else:
    run_time_iter = np.inf


dealias = 3/2
dtype = np.float64

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

scale_ratio = nz_sol/nz
b0.change_scales(scale_ratio)
q0.change_scales(scale_ratio)
logger.info('rescaling b0, q0 to match background from {:} to {:} coeffs (ratio: {:})'.format(nz, nz_sol, scale_ratio))

b0['g'] = sol['b']
q0['g'] = sol['q']

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

T = b - β*z_grid
qs = np.exp(α*T)
rh = q*np.exp(-α*T)

ΔT = -1
q_surface = dist.Field(name='q_surface')
if q0['g'].size > 0 :
    q_surface['g'] = q0(z=0).evaluate()['g']

grad = lambda A: de.Gradient(A, coords)
trans = lambda A: de.TransposeComponents(A)
curl = lambda A: de.Curl(A)

e = grad(u) + trans(grad(u))
ω = curl(u)

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


problem.add_equation('div(u) + τp + 1/PdR*dot(lift(τu2,-1),ez) = 0')
problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b*ez + lift(τu1, -1) + lift(τu2, -2) = cross(u, ω)')
# problem.add_equation('dt(b) - P*lap(b) + u@grad(b0) - γ/tau*(q-α*qs0*b)*scrN + lift(τb1, -1) + lift(τb2, -2) = - (u@grad(b)) + γ/tau*((q-qs)*H(q-qs) - (q-α*qs0*b)*scrN_g)')
# problem.add_equation('dt(q) - S*lap(q) + u@grad(q0) + 1/tau*(q-α*qs0*b)*scrN + lift(τq1, -1) + lift(τq2, -2) = - (u@grad(q)) - 1/tau*((q-qs)*H(q-qs) - (q-α*qs0*b)*scrN_g)')
problem.add_equation('dt(b) - P*lap(b) + lift(τb1, -1) + lift(τb2, -2) = - (u@grad(b)) + γ/tau*((q-qs)*H(q-qs))')
problem.add_equation('dt(q) - S*lap(q) + lift(τq1, -1) + lift(τq2, -2) = - (u@grad(q)) - 1/tau*((q-qs)*H(q-qs))')
problem.add_equation('b(z=0) = 0')
problem.add_equation('b(z=Lz) = β + ΔT') # technically β*Lz
problem.add_equation('q(z=0) = q_surface*qs(z=0)')
problem.add_equation('q(z=Lz) = np.exp(α*ΔT)')
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
noise.low_pass_filter(scales=0.75)

# noise ICs in moisture
b0.change_scales(1)
q0.change_scales(1)
b['g'] = b0['g']
q['g'] = q0['g']
b['g'] += noise['g']*np.cos(np.pi/2*z/Lz)

ts = de.SBDF2
cfl_safety_factor = 0.2

solver = problem.build_solver(ts)
solver.stop_sim_time = run_time_buoy
solver.stop_iteration = run_time_iter

Δt = max_Δt = min(float(args['--max_dt']), tau/4)
logger.info('setting Δt = min(--max_dt={:.2g}, tau/4={:.2g})'.format(float(args['--max_dt']), tau/4))
cfl = flow_tools.CFL(solver, Δt, safety=cfl_safety_factor, cadence=1, threshold=0.1,
                      max_change=1.5, min_change=0.5, max_dt=max_Δt)
cfl.add_velocity(u)

report_cadence = 1e2

integ = lambda A: de.Integrate(de.Integrate(A, 'x'), 'z')
avg = lambda A: integ(A)/(Lx*Lz)
x_avg = lambda A: de.Integrate(A, 'x')/(Lx)

Re = np.sqrt(u@u)/PdR
KE = 0.5*u@u
PE = PtR*b
QE = PtR*γ*q
ME = PE + QE # moist static energy
Q_eq = (q-qs)*H(q - qs)

snapshots = solver.evaluator.add_file_handler(data_dir+'/snapshots', sim_dt=2, max_writes=20)
snapshots.add_task(b, name='b')
snapshots.add_task(q, name='q')
snapshots.add_task(b-x_avg(b), name='b_fluc')
snapshots.add_task(q-x_avg(q), name='q_fluc')
snapshots.add_task(rh, name='rh')
snapshots.add_task(rh-x_avg(rh), name='rh_fluc')
snapshots.add_task(ex@u, name='ux')
snapshots.add_task(ez@u, name='uz')
snapshots.add_task(ey@ω, name='vorticity')
snapshots.add_task(ω@ω, name='enstrophy')
snapshots.add_task(x_avg(b), name='b_avg')
snapshots.add_task(x_avg(q), name='q_avg')
snapshots.add_task(x_avg(b+γ*q), name='m_avg')
snapshots.add_task(x_avg(rh), name='rh_avg')
snapshots.add_task(x_avg(Q_eq), name='Q_eq_avg')
snapshots.add_task(x_avg(ez@u*q), name='uq_avg')
snapshots.add_task(x_avg(ez@u*b), name='ub_avg')
snapshots.add_task(x_avg(ex@u), name='ux_avg')
snapshots.add_task(x_avg(ez@u), name='uz_avg')
snapshots.add_task(x_avg(np.sqrt((u-x_avg(u))@(u-x_avg(u)))), name='u_rms')


trace_dt = 0.5
traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=trace_dt, max_writes=None)
traces.add_task(avg(KE), name='KE')
traces.add_task(avg(PE), name='PE')
traces.add_task(avg(QE), name='QE')
traces.add_task(avg(ME), name='ME')
traces.add_task(avg(Q_eq), name='Q_eq')
traces.add_task(avg(Re), name='Re')
traces.add_task(avg(ω@ω), name='enstrophy')
traces.add_task(x_avg(np.sqrt(τu1@τu1)), name='τu1')
traces.add_task(x_avg(np.sqrt(τu2@τu2)), name='τu2')
traces.add_task(x_avg(np.abs(τb1)), name='τb1')
traces.add_task(x_avg(np.abs(τb2)), name='τb2')
traces.add_task(x_avg(np.abs(τq1)), name='τq1')
traces.add_task(x_avg(np.abs(τq2)), name='τq2')
traces.add_task(np.abs(τp), name='τp')


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
try:
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
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
