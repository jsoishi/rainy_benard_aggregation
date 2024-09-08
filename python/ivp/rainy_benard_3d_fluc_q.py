"""
Dedalus script for calculating dynamic solutions to the Rainy-Benard system.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves IVPs, starting from an analytic base state.

Usage:
    rainy_benard.py [options]

Options:
                      Properties of analytic atmosphere
    --alpha=<alpha>   alpha value [default: 3]
    --beta=<beta>     beta value  [default: 1.1]
    --gamma=<gamma>   gamma value [default: 0.19]
    --q0=<q0>         basal q value [default: 0.6]

    --Rayleigh=<Ra>   Rayleigh number [default: 1e5]

    --aspect=<a>      Aspect ratio of domain, Lx/Lz [default: 8]

    --tau=<tau>       Timescale for moisture reaction [default: 1e-2]
    --k=<k>           Sharpness of smooth phase transition [default: 1e5]

    --erf             Use an erf for the phase transition (default)
    --tanh            Use a tanh for the phase transition

    --Legendre        Use Legendre polynomials (default)
    --Chebyshev       Use Chebyshev polynomials

    --nondim=<n>      Non-Nondimensionalization [default: buoyancy]

    --top-stress-free     Stress-free upper boundary (default)
    --stress-free         Stress-free both boundaries
    --no-slip             No slip both boundaries

    --nz=<nz>         Number z coeffs to use in IVP
    --nx=<nx>         Number of x and y coeffs to use in IVP; if not set, scales nz by aspect

    --mesh=<mesh>     Processor mesh for 3-D runs; if not set a sensible guess will be made

    --max_dt=<dt>     Largest timestep to use; should be set by oscillation timescales of waves (Brunt) [default: 1]

    --implicit        Enable implicit treatment of condensation term

    --run_time_diff=<rtd>      Run time, in diffusion times [default: 1]
    --run_time_buoy=<rtb>      Run time, in buoyancy times
    --run_time_iter=<rti>      Run time, number of iterations; if not set, n_iter=np.inf

    --output_dt=<output_dt>    Delta T for outputs (in buoyancy times) [default: 5]

    --no-output       Suppress disk writing output, for timing

    --full_case_label  Use a longer form of case labelling including erf/tanh, Tz/L in name

    --timestepper=<ts>         Timestepping algorithm to use [default: SBDF4]

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
Ly = Lx

from mpi4py import MPI
nproc = MPI.COMM_WORLD.size
mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(nproc)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))


coords = de.CartesianCoordinates('y', 'x', 'z', right_handed=False)
dist = de.Distributor(coords, mesh=mesh, dtype=dtype)

import os
import analytic_atmosphere

from analytic_zc import f_zc as zc_analytic
from analytic_zc import f_Tc as Tc_analytic
α = float(args['--alpha'])
β = float(args['--beta'])
γ = float(args['--gamma'])
k = float(args['--k'])
q0_basal = float(args['--q0'])
tau = float(args['--tau'])

if q0_basal < 1:
    atm_name = 'unsaturated'
elif q0_basal == 1:
    atm_name = 'saturated'
else:
    raise ValueError("q0 has invalid value, q0 = {:}".format(q0_basal))

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

    sol = sol(dist, zb, β, γ, zc, Tc, dealias=dealias, q0=q0_basal, α=α)
elif atm_name == 'saturated':
    sol = analytic_atmosphere.saturated
    sol = sol(dist, zb, β, γ, dealias=dealias, q0=q0_basal, α=α)

sol['b'].change_scales(1)
sol['q'].change_scales(1)
sol['z'].change_scales(1)
sol['b'] = sol['b']['g']
sol['q'] = sol['q']['g']

if not os.path.exists('{:s}/'.format(case)) and dist.comm.rank == 0:
    os.makedirs('{:s}/'.format(case))

if args['--nx']:
    nx = int(float(args['--nx']))
else:
    nx = int(aspect*nz)
ny = nx

data_dir = case+'/rainy_benard_Ra{:}_tau{:.2g}_k{:.2g}_nz{:d}_nx{:d}_ny{:d}_a{:}'.format(args['--Rayleigh'], tau, k, nz, nx, ny, args['--aspect'])

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
logger.info('α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(α,β,γ,tau,k))

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
yb = de.RealFourier(coords['y'], size=ny, bounds=(0, Ly), dealias=dealias)
x = dist.local_grid(xb)
y = dist.local_grid(yb)
z = dist.local_grid(zb)

b0 = dist.Field(name='b0', bases=zb)
q0 = dist.Field(name='q0', bases=zb)

b0.change_scales(1)
q0.change_scales(1)

# apply analytic solution
b0['g'] = sol['b']
q0['g'] = sol['q']

bases = (yb, xb, zb)
bases_h = (yb, xb)

p = dist.Field(name='p', bases=bases)
u = dist.VectorField(coords, name='u', bases=bases)
b1 = dist.Field(name='b1', bases=bases)
q1 = dist.Field(name='q1', bases=bases)

b = b1 + b0
q = q1 + q0

τc0 = dist.Field(name='τc0')
τc1 = dist.Field(name='τc1')
τu1 = dist.VectorField(coords, name='τu1', bases=bases_h)
τu2 = dist.VectorField(coords, name='τu2', bases=bases_h)
τb1 = dist.Field(name='τb1', bases=bases_h)
τb2 = dist.Field(name='τb2', bases=bases_h)
τq1 = dist.Field(name='τq1', bases=bases_h)
τq2 = dist.Field(name='τq2', bases=bases_h)

ey, ex, ez = coords.unit_vector_fields(dist)

from scipy.special import erf
if args['--tanh']:
    H = lambda A: 0.5*(1+np.tanh(k*A))
else:
    H = lambda A: 0.5*(1+erf(k*A))

z_grid = dist.Field(name='z_grid', bases=zb)
z_grid['g'] = z

T = b - β*z_grid
T0 = b0 - β*z_grid
qs = np.exp(α*T)
qs0 = np.exp(α*T0)
rh = q*np.exp(-α*T)

scrN = H(q0 - qs0).evaluate()
scrN.name = 'scrN'

ΔT = -1
Δm = (β-1) + γ*(np.exp(α*ΔT)-q0_basal)

grad_q0 = de.grad(q0).evaluate()
grad_b0 = de.grad(b0).evaluate()

ncc_cutoff = 1e-10
ncc_list = [grad_b0, grad_q0, scrN]
if args['--implicit']:
    ncc_list += [qs0]
for ncc in ncc_list:
    logger.info("{}: {}".format(ncc.evaluate(), np.where(np.abs(ncc.evaluate()['c']) >= ncc_cutoff)[0].shape))

e_ij = de.grad(u) + de.trans(de.grad(u))
ω = de.curl(u)

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

lift_basis1 = zb.derivative_basis(1)
lift1 = lambda A, n: de.Lift(A, lift_basis1, n)

lift_basis2 = zb.derivative_basis(2)
lift2 = lambda A, n: de.Lift(A, lift_basis2, n)

τ_d = lift1(τc1, -1) + τc0
τ_u = lift2(τu1, -1) + lift2(τu2, -2)
τ_b = lift2(τb1, -1) + lift2(τb2, -2)
τ_q = lift2(τq1, -1) + lift2(τq2, -2)

taus = [τc0, τc1, τu1, τu2, τb1, τb2, τq1, τq2]
vars = [p, u, b1, q1]
problem = de.IVP(vars+taus, namespace=locals())
problem.add_equation('div(u) + τ_d = 0')
problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b1*ez + τ_u = cross(u, ω)')
if args['--implicit']:

    problem.add_equation('dt(b1) - P*lap(b1) + u@grad_b0 - γ/tau*(q1-α*qs0*b1)*scrN + τ_b = - (u@grad(b1)) + γ/tau*((q-qs)*H(q-qs)) - γ/tau*(q1-α*qs0*b1)*scrN + P*lap(b0)')
    problem.add_equation('dt(q1) - S*lap(q1) + u@grad_q0 + 1/tau*(q1-α*qs0*b1)*scrN + τ_q = - (u@grad(q1)) - 1/tau*((q-qs)*H(q-qs)) + 1/tau*(q1-α*qs0*b1)*scrN + S*lap(q0)')
else:
    problem.add_equation('dt(b1) - P*lap(b1) + u@grad_b0 + τ_b = - (u@grad(b1)) + γ/tau*((q-qs)*H(q-qs)) + P*lap(b0)')
    problem.add_equation('dt(q1) - S*lap(q1) + u@grad_q0 + τ_q = - (u@grad(q1)) - 1/tau*((q-qs)*H(q-qs)) + S*lap(q0)')
problem.add_equation('b1(z=0) = 0')
problem.add_equation('b1(z=Lz) = 0')
problem.add_equation('q1(z=0) = 0')
problem.add_equation('q1(z=Lz) = 0')
problem.add_equation('u(z=0) = 0')
# stress-free top
problem.add_equation('ez@(ex@e_ij(z=Lz)) = 0')
problem.add_equation('ez@(ey@e_ij(z=Lz)) = 0')
problem.add_equation('ez@u(z=Lz) = 0')
problem.add_equation('integ(ez@τu2) = 0')
problem.add_equation('integ(p) = 0')

# initial conditions
amp = np.abs(Δm)*1e-3
noise = dist.Field(name='noise', bases=bases)
noise.fill_random('g', seed=42, distribution='normal', scale=amp) # Random noise
noise.low_pass_filter(scales=0.75)

# noise ICs in buoyancy
b1['g'] += noise['g']*np.sin(np.pi*z/Lz)

if args['--timestepper'] == 'SBDF4':
    ts = de.SBDF4
elif args['--timestepper'] == 'SBDF2':
    ts = de.SBDF2
elif args['--timestepper'] == 'RK222':
    ts = de.RK222
elif args['--timestepper'] == 'RK443':
    ts = de.RK443
else:
    raise ValueError(f'timestepper {args["--timestepper"]} not currently implemented in this script')
cfl_safety_factor = 0.2

solver = problem.build_solver(ts, ncc_cutoff=ncc_cutoff, enforce_real_cadence=np.inf)
solver.stop_sim_time = run_time_buoy
solver.stop_iteration = run_time_iter

if args['--implicit']:
    Δt = max_Δt = float(args['--max_dt'])
    logger.info('setting Δt = --max_dt={:.2g}, implicitly stepping over tau={:.2g})'.format(float(args['--max_dt']), tau))
else:
    Δt = max_Δt = min(float(args['--max_dt']), tau/4)
    logger.info('setting Δt = min(--max_dt={:.2g}, tau/4={:.2g}), explicitly capturing tau'.format(float(args['--max_dt']), tau/4))
cfl = flow_tools.CFL(solver, Δt, safety=cfl_safety_factor, cadence=1, threshold=0.1,
                      max_change=1.5, min_change=0.5, max_dt=max_Δt)
cfl.add_velocity(u)

vol = Lx*Ly*Lz
avg = lambda A: de.Integrate(A)/vol
h_avg = lambda A: de.Integrate(de.Integrate(A, 'x'), 'y')/(Lx*Ly)

Re = np.sqrt(u@u)/PdR
KE = 0.5*u@u
PE = PtR*b
QE = PtR*γ*q
ME = PE + QE # moist static energy
Q_eq = (q-qs)*H(q - qs)
m = b+γ*q

if not args['--no-output']:
    snap_dt = float(args['--output_dt'])
    snapshots = solver.evaluator.add_file_handler(data_dir+'/snapshots', sim_dt=snap_dt, max_writes=20)
    snapshots.add_task(b(y=Ly/2), name='b 0.5 y')
    snapshots.add_task(q(y=Ly/2), name='q 0.5 y')
    snapshots.add_task(m(y=Ly/2), name='m 0.5 y')
    snapshots.add_task(rh(y=Ly/2), name='rh 0.5 y')
    snapshots.add_task((b-h_avg(b))(y=Ly/2), name='b_fluc 0.5 y')
    snapshots.add_task((q-h_avg(q))(y=Ly/2), name='q_fluc 0.5 y')
    snapshots.add_task((m-h_avg(m))(y=Ly/2), name='m_fluc 0.5 y')
    snapshots.add_task((rh-h_avg(rh))(y=Ly/2), name='rh_fluc 0.5 y')
    snapshots.add_task(ex@u(y=Ly/2), name='ux 0.5 y')
    snapshots.add_task(ez@u(y=Ly/2), name='uz 0.5 y')
    snapshots.add_task(ey@ω(y=Ly/2), name='vorticity y 0.5 y')
    snapshots.add_task((ω@ω)(y=Ly/2), name='enstrophy 0.5 y')
    for z_frac in [0.25, 0.5, 0.75]:
        snapshots.add_task(b(z=Lz*z_frac), name=f'b {z_frac} z')
        snapshots.add_task(q(z=Lz*z_frac), name=f'q {z_frac} z')
        snapshots.add_task(m(z=Lz*z_frac), name=f'm {z_frac} z')
        snapshots.add_task(rh(z=Lz*z_frac), name=f'rh {z_frac} z')
        snapshots.add_task(ex@u(z=Lz*z_frac), name=f'ux {z_frac} z')
        snapshots.add_task(ez@u(z=Lz*z_frac), name=f'uz {z_frac} z')
        snapshots.add_task(ez@ω(z=Lz*z_frac), name=f'vorticity z {z_frac} z')
        snapshots.add_task((ω@ω)(z=Lz*z_frac), name=f'enstrophy {z_frac} z')

    averages = solver.evaluator.add_file_handler(data_dir+'/averages', sim_dt=snap_dt, max_writes=None)
    averages.add_task(h_avg(b), name='b')
    averages.add_task(h_avg(q), name='q')
    averages.add_task(h_avg(m), name='m')
    averages.add_task(h_avg(rh), name='rh')
    averages.add_task(h_avg(Q_eq), name='Q_eq')
    averages.add_task(h_avg(ez@u*q), name='uq')
    averages.add_task(h_avg(ez@u*b), name='ub')
    averages.add_task(h_avg(ey@u), name='uy')
    averages.add_task(h_avg(ex@u), name='ux')
    averages.add_task(h_avg(ez@u), name='uz')
    averages.add_task(h_avg(np.sqrt((u-h_avg(u))@(u-h_avg(u)))), name='u_rms')
    averages.add_task(h_avg(ω@ω), name='enstrophy')
    averages.add_task(h_avg((ω-h_avg(ω))@(ω-h_avg(ω))), name='enstrophy_rms')

    slices = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt=snap_dt, max_writes=None)
    slices.add_task(b(z=0.5, y=0.5), name='b')
    slices.add_task(q(z=0.5, y=0.5), name='q')
    slices.add_task(m(z=0.5, y=0.5), name='m')
    slices.add_task(rh(z=0.5, y=0.5), name='rh')
    slices.add_task(Q_eq(z=0.5, y=0.5), name='Q_eq')
    slices.add_task((ez@u*q)(z=0.5, y=0.5), name='uq')
    slices.add_task((ez@u*b)(z=0.5, y=0.5), name='ub')
    slices.add_task((ez@u)(z=0.5, y=0.5), name='uz')
    slices.add_task((np.sqrt((u-h_avg(u))@(u-h_avg(u))))(z=0.5, y=0.5), name='u_rms')
    slices.add_task((ω@ω)(z=0.5, y=0.5), name='enstrophy')
    slices.add_task(((ω-h_avg(ω))@(ω-h_avg(ω)))(z=0.5, y=0.5), name='enstrophy_rms')

    scalar_dt = snap_dt/5
    scalars = solver.evaluator.add_file_handler(data_dir+'/scalars', sim_dt=scalar_dt, max_writes=None)
    scalars.add_task(avg(KE), name='KE')
    scalars.add_task(avg(PE), name='PE')
    scalars.add_task(avg(QE), name='QE')
    scalars.add_task(avg(ME), name='ME')
    scalars.add_task(avg(Q_eq), name='Q_eq')
    scalars.add_task(avg(Re), name='Re')
    scalars.add_task(avg(ω@ω), name='enstrophy')
    scalars.add_task(avg(np.abs(de.div(u))), name='div_u')
    scalars.add_task(h_avg(np.sqrt(τu1@τu1)), name='τu1')
    scalars.add_task(h_avg(np.sqrt(τu2@τu2)), name='τu2')
    scalars.add_task(h_avg(np.abs(τb1)), name='τb1')
    scalars.add_task(h_avg(np.abs(τb2)), name='τb2')
    scalars.add_task(h_avg(np.abs(τq1)), name='τq1')
    scalars.add_task(h_avg(np.abs(τq2)), name='τq2')
    scalars.add_task(np.abs(τc1), name='τc1')
    scalars.add_task(np.abs(τc0), name='τc0')
    scalars.add_task(np.sqrt(avg(τ_d**2)), name='τ_d')
    scalars.add_task(np.sqrt(avg(τ_q**2)), name='τ_q')
    scalars.add_task(np.sqrt(avg(τ_b**2)), name='τ_b')
    scalars.add_task(np.sqrt(avg(τ_u@τ_u)), name='τ_u')

    checkpoint_wall_dt = 3.9*3600 # trigger slightly before a 4 hour interval
    checkpoints = solver.evaluator.add_file_handler(data_dir+'/checkpoints', wall_dt=checkpoint_wall_dt, max_writes=1)
    #checkpoints.add_system(solver.state)
    checkpoints.add_task(p, layout='c')
    checkpoints.add_task(b, layout='c')
    checkpoints.add_task(q, layout='c')
    checkpoints.add_task(u, layout='c')

report_cadence = 1e2
flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(Re, name='Re')
flow.add_property(KE, name='KE')
flow.add_property(np.abs(de.div(u)), name='div_u')
flow.add_property(np.sqrt(τ_d**2)+np.sqrt(τ_u@τ_u)+np.sqrt(τ_b**2)+np.sqrt(τ_q**2), name='|taus|')
flow.add_property(np.sqrt(avg(τ_d**2)), name='|τ_d|')
flow.add_property(np.sqrt(avg(τ_q**2)), name='|τ_q|')
flow.add_property(np.sqrt(avg(τ_b**2)), name='|τ_b|')
flow.add_property(np.sqrt(avg(τ_u@τ_u)), name='|τ_u|')

good_solution = True
KE_avg = 0
try:
    while solver.proceed and good_solution:
        # advance
        solver.step(Δt)
        if solver.iteration % report_cadence == 0:
            τ_max = np.max([flow.max('|τ_u|'),flow.max('|τ_b|'),flow.max('|τ_q|'),flow.max('|τ_d|')])
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
