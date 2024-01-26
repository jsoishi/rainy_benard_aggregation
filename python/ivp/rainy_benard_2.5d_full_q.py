"""
Dedalus script for calculating dynamic solutions to the Rainy-Benard system.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves IVPs for an existing atmospheres, solved for by scripts in the nlbvp section.

Usage:
    rainy_benard.py [options]

Options:
                      Properties of analytic atmosphere
    --alpha=<alpha>   alpha value [default: 3]
    --beta=<beta>     beta value  [default: 1.1]
    --gamma=<gamma>   gamma value [default: 0.19]
    --q0=<q0>         basal q value [default: 0.6]

    --Rayleigh=<Ra>   Rayleigh number [default: 1e5]

    --aspect=<a>      Aspect ratio of domain, Lx/Lz [default: 10]

    --tau=<tau>       Timescale for moisture reaction [default: 0.1]
    --k=<k>           Sharpness of smooth phase transition [default: 1e4]

    --erf             Use an erf for the phase transition (default)
    --tanh            Use a tanh for the phase transition

    --Legendre        Use Legendre polynomials (default)
    --Chebyshev       Use Chebyshev polynomials

    --nondim=<n>      Non-Nondimensionalization [default: buoyancy]

    --top-stress-free     Stress-free upper boundary (default)
    --stress-free         Stress-free both boundaries
    --no-slip             No slip both boundaries

    --nz=<nz>         Number z coeffs to use in IVP
    --nx=<nx>         Number of x coeffs to use in IVP; if not set, scales nz by aspect

    --max_dt=<dt>     Largest timestep to use; should be set by oscillation timescales of waves (Brunt) [default: 1]

    --run_time_diff=<rtd>      Run time, in diffusion times [default: 1]
    --run_time_buoy=<rtb>      Run time, in buoyancy times
    --run_time_iter=<rti>      Run time, number of iterations; if not set, n_iter=np.inf

    --no-output       Suppress disk writing output, for timing

    --full_case_label  Use a longer form of case labelling including erf/tanh, Tz/L in name

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

import dedalus.public as de
from dedalus.extras import flow_tools

aspect = float(args['--aspect'])

dealias = 3/2
dtype = np.float64

Lz = 1
Lx = aspect

from mpi4py import MPI
nproc = MPI.COMM_WORLD.size

coords = de.CartesianCoordinates('y', 'x', 'z', right_handed=False)
dist = de.Distributor(coords, mesh=[1,nproc], dtype=dtype)

import os
import analytic_atmosphere

from analytic_zc import f_zc as zc_analytic
from analytic_zc import f_Tc as Tc_analytic
α = float(args['--alpha'])
β = float(args['--beta'])
γ = float(args['--gamma'])
k = float(args['--k'])
q0 = float(args['--q0'])
tau = float(args['--tau'])

if q0 < 1:
    atm_name = 'unsaturated'
elif q0 == 1:
    atm_name = 'saturated'
else:
    raise ValueError("q0 has invalid value, q0 = {:}".format(q0))

case = 'analytic'
case += '_{:s}/alpha{:}_beta{:}_gamma{:}_q{:}'.format(atm_name, args['--alpha'],args['--beta'],args['--gamma'], args['--q0'])

nz = int(float(args['--nz']))
if args['--Chebyshev']:
    zb = de.ChebyshevT(coords['z'], size=nz, bounds=(0, Lz), dealias=dealias)
else:
    zb = de.Legendre(coords['z'], size=nz, bounds=(0, Lz), dealias=dealias)

if atm_name == 'unsaturated':
    sol = analytic_atmosphere.unsaturated
    zc = zc_analytic()(γ)
    Tc = Tc_analytic()(γ)

    sol = sol(dist, zb, β, γ, zc, Tc, dealias=dealias, q0=q0, α=α)
elif atm_name == 'saturated':
    sol = analytic_atmosphere.saturated
    sol = sol(dist, zb, β, γ, dealias=dealias, q0=q0, α=α)

sol['b'].change_scales(1)
sol['q'].change_scales(1)
sol['b'] = sol['b']['g']
sol['q'] = sol['q']['g']
sol['z'].change_scales(1)
nz_sol = sol['z']['g'].shape[-1]
if not os.path.exists('{:s}/'.format(case)) and dist.comm.rank == 0:
    os.makedirs('{:s}/'.format(case))

if args['--nx']:
    nx = int(float(args['--nx']))
else:
    nx = int(aspect)*nz

data_dir = case+'/rainy_benard_Ra{:}_tau{:.2g}_k{:.2g}_nz{:d}_nx{:d}'.format(args['--Rayleigh'], tau, k, nz, nx)

if args['--full_case_label']:
    if args['--tanh']:
        case += '_tanh'
    else:
        case += '_erf'
    if args['--Chebyshev']:
        case += '_Tz'
    else:
        case += '_L'

if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

logger.info('saving data to: {:}'.format(data_dir))
logger.info('α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(α,β,γ,tau, k))

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

xb = de.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=dealias)
x = dist.local_grid(xb)
z = dist.local_grid(zb)

b0 = dist.Field(name='b0', bases=zb)
q0 = dist.Field(name='q0', bases=zb)

scale_ratio = 1 #nz_sol/nz
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

ey, ex, ez = coords.unit_vector_fields(dist)

from scipy.special import erf
if args['--tanh']:
    H = lambda A: 0.5*(1+np.tanh(k*A))
else:
    H = lambda A: 0.5*(1+erf(k*A))

z_grid = dist.Field(name='z_grid', bases=zb)
z_grid['g'] = z

T = b - β*z_grid
qs = np.exp(α*T)
rh = q*np.exp(-α*T)

ΔT = -1
q_surface = dist.Field(name='q_surface')
if q0['g'].size > 0 :
    q_surface['g'] = q0(z=0).evaluate()['g']

div = lambda A: de.Divergence(A)
grad = lambda A: de.Gradient(A, coords)
trans = lambda A: de.TransposeComponents(A)
curl = lambda A: de.Curl(A)

e = grad(u) + trans(grad(u))
ω = curl(u)

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

# zb1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
# zb2 = zb.clone_with(a=zb.a+2, b=zb.b+2)
# lift1 = lambda A, n: de.Lift(A, zb1, n)
# lift = lambda A, n: de.Lift(A, zb2, n)

lift_basis = zb.derivative_basis(1)
lift = lambda A: de.Lift(A, lift_basis, -1)

lift_basis2 = zb.derivative_basis(2)
lift2 = lambda A: de.Lift(A, lift_basis2, -1)
lift2_2 = lambda A: de.Lift(A, lift_basis2, -2)

vars = [p, u, b, q, τp, τu1, τu2, τb1, τb2, τq1, τq2]
problem = de.IVP(vars, namespace=locals())
problem.add_equation('div(u) + lift(τp) = 0')
problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b*ez + lift2_2(τu1) + lift2(τu2) = cross(u, ω)')
problem.add_equation('dt(b) - P*lap(b) + lift2_2(τb1) + lift2(τb2) = - (u@grad(b)) + γ/tau*((q-qs)*H(q-qs))')
problem.add_equation('dt(q) - S*lap(q) + lift2_2(τq1) + lift(τq2) = - (u@grad(q)) - 1/tau*((q-qs)*H(q-qs))')
problem.add_equation('b(z=0) = 0')
problem.add_equation('b(z=Lz) = β + ΔT') # technically β*Lz
problem.add_equation('q(z=0) = q_surface*qs(z=0)')
problem.add_equation('q(z=Lz) = np.exp(α*ΔT)')
if args['--stress-free']:
    logger.info("bottom is stress-free")
    problem.add_equation('ez@u(z=0) = 0', condition="nx!=0")
    problem.add_equation('ez@(ex@e(z=0)) = 0')
    problem.add_equation('ez@(ey@e(z=0)) = 0')
    problem.add_equation("ez@τu1 = 0", condition="nx==0")
else:
    logger.info("bottom is no-slip")
    problem.add_equation("u(z=0) = 0", condition="nx!=0")
    problem.add_equation("ex@u(z=0) = 0", condition="nx==0")
    problem.add_equation("ey@u(z=0) = 0", condition="nx==0")
    problem.add_equation("ez@τu1 = 0", condition="nx==0")
if not args['--no-slip'] or args['--stress-free']:
    logger.info("top is stress-free")
    problem.add_equation('ez@u(z=Lz) = 0')
    problem.add_equation('ez@(ex@e(z=Lz)) = 0')
    problem.add_equation('ez@(ey@e(z=Lz)) = 0')
else:
    logger.info("top is no-slip")
    problem.add_equation('u(z=Lz) = 0')
problem.add_equation('integ(p) = 0')

# initial conditions
amp = 1e-4

noise = dist.Field(name='noise', bases=bases)
noise.fill_random('g', seed=42, distribution='normal', scale=amp) # Random noise
noise.low_pass_filter(scales=0.75)

# noise ICs in buoyancy
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

vol = Lx*Lz
integ = lambda A: de.Integrate(de.Integrate(A, 'x'), 'z')
avg = lambda A: integ(A)/vol
x_avg = lambda A: de.Integrate(A, 'x')/(Lx)

Re = np.sqrt(u@u)/PdR
KE = 0.5*u@u
PE = PtR*b
QE = PtR*γ*q
ME = PE + QE # moist static energy
Q_eq = (q-qs)*H(q - qs)
m = b+γ*q

if not args['--no-output']:
    snap_dt = 5
    snapshots = solver.evaluator.add_file_handler(data_dir+'/snapshots', sim_dt=snap_dt, max_writes=20)
    snapshots.add_task(b, name='b')
    snapshots.add_task(q, name='q')
    snapshots.add_task(m, name='m')
    snapshots.add_task(rh, name='rh')
    snapshots.add_task(Q_eq, name='c')
    snapshots.add_task(b-x_avg(b), name='b_fluc')
    snapshots.add_task(q-x_avg(q), name='q_fluc')
    snapshots.add_task(m-x_avg(m), name='m_fluc')
    snapshots.add_task(rh-x_avg(rh), name='rh_fluc')
    snapshots.add_task(ex@u, name='ux')
    snapshots.add_task(ez@u, name='uz')
    snapshots.add_task(ey@ω, name='vorticity')
    snapshots.add_task(ω@ω, name='enstrophy')

    averages = solver.evaluator.add_file_handler(data_dir+'/averages', sim_dt=snap_dt, max_writes=None)
    averages.add_task(x_avg(b), name='b')
    averages.add_task(x_avg(q), name='q')
    averages.add_task(x_avg(m), name='m')
    averages.add_task(x_avg(rh), name='rh')
    averages.add_task(x_avg(Q_eq), name='Q_eq')
    averages.add_task(x_avg(ez@u*q), name='uq')
    averages.add_task(x_avg(ez@u*b), name='ub')
    averages.add_task(x_avg(ex@u), name='ux')
    averages.add_task(x_avg(ez@u), name='uz')
    averages.add_task(x_avg(np.sqrt((u-x_avg(u))@(u-x_avg(u)))), name='u_rms')
    averages.add_task(x_avg(ω@ω), name='enstrophy')
    averages.add_task(x_avg((ω-x_avg(ω))@(ω-x_avg(ω))), name='enstrophy_rms')

    slice = solver.evaluator.add_file_handler(data_dir+'/slice', sim_dt=snap_dt, max_writes=None)
    slice.add_task(b(z=0.5), name='b')
    slice.add_task(q(z=0.5), name='q')
    slice.add_task(m(z=0.5), name='m')
    slice.add_task(rh(z=0.5), name='rh')
    slice.add_task(Q_eq(z=0.5), name='Q_eq')
    slice.add_task((ez@u*q)(z=0.5), name='uq')
    slice.add_task((ez@u*b)(z=0.5), name='ub')
    slice.add_task((ez@u)(z=0.5), name='uz')
    slice.add_task((np.sqrt((u-x_avg(u))@(u-x_avg(u))))(z=0.5), name='u_rms')
    slice.add_task((ω@ω)(z=0.5), name='enstrophy')
    slice.add_task(((ω-x_avg(ω))@(ω-x_avg(ω)))(z=0.5), name='enstrophy_rms')

    trace_dt = snap_dt/5
    traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=trace_dt, max_writes=None)
    traces.add_task(avg(KE), name='KE')
    traces.add_task(avg(PE), name='PE')
    traces.add_task(avg(QE), name='QE')
    traces.add_task(avg(ME), name='ME')
    traces.add_task(avg(Q_eq), name='Q_eq')
    traces.add_task(avg(Re), name='Re')
    traces.add_task(avg(ω@ω), name='enstrophy')
    traces.add_task(avg(np.abs(div(u))), name='div_u')
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
flow.add_property(np.abs(div(u)), name='div_u')
flow.add_property(np.sqrt(τu1@τu1), name='|τu1|')
flow.add_property(np.sqrt(τu2@τu2), name='|τu2|')
flow.add_property(np.abs(τb1), name='|τb1|')
flow.add_property(np.abs(τb2), name='|τb2|')
flow.add_property(np.abs(τq1), name='|τq1|')
flow.add_property(np.abs(τq2), name='|τq2|')
flow.add_property(np.abs(τp), name='|τp|')

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
            div_u_avg = flow.volume_integral('div_u')/vol
            div_u_max = flow.max('div_u')
            log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:5.1e}'.format(solver.iteration, solver.sim_time, Δt)
            log_string += ', KE: {:.2g}, Re: {:.2g} ({:.2g})'.format(KE_avg, Re_avg, Re_max)
            log_string += ', div(u): {:.2g}'.format(div_u_max)
            log_string += ', τ: {:.2g}'.format(τ_max)
            logger.info(log_string)
        Δt = cfl.compute_timestep()
        good_solution = np.isfinite(Δt)*np.isfinite(KE_avg)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
