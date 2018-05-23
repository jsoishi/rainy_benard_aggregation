"""
Dedalus script for Rainy Benard Aggregation study

Simulation script for 3D Rayleigh-Benard Moist convection.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `process.py` script in this
folder can be used to merge distributed save files from parallel runs and plot
the snapshots from the command line.

The scripts make three different types of outputs: full 3D data dumps, 2D slices
through the data cube, and 1D horizontally averaged profiles.  The `process.py`
script can only plot the 2D slices.

To run, join, and plot vertical slices using 4 processes, for instance, you could
use:
$ mpiexec -n 4 python3 rayleigh_benard.py
$ mpiexec -n 4 python3 process.py join slices
$ mpiexec -n 4 python3 process.py plot_slices slices/*.h5 --type=horizontal
On four processors, the simulation runs in a couple of minutes.

This script is currently parallelized using a 1D mesh decomposition, so that the 
maximum number of processors is set by the direction with the lowest resolution.
A 2D mesh decomposition can be used to run the problem with more processors.  To
do this, you must specify mesh=[N1,N2] in the domain creation, where N1*N2 is the
total number of processors.  In grid space, each processor has all the x data,
and a fraction of the y data (N1/Ny), and a fraction of the z data (N2/Nz).

Default paramters from Vallis, Parker, and Tobias (2018)
http://empslocal.ex.ac.uk/people/staff/gv219/papers/VPT_convection18.pdf

Usage:
    moistrb.py [--beta=<beta> --Rayleigh=<Rayleigh> --Prandtl=<Prandtl> --Prandtlm=<Prandtlm> --F=<F> --alpha=<alpha> --gamma=<gamma> --DeltaT=<DeltaT> --sigma2=<sigma2> --q0=<q0> --nx=<nx> --ny=<ny> --nz=<nz> --Lx=<Lx> --Ly=<Ly> --Lz=<Lz> --restart=<restart_file> --filter=<filter> --mesh=<mesh> --nondim=<nondim>] 

Options:
    --Rayleigh=<Rayleigh>    Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>      Prandtl number [default: 1]
    --Prandtlm=<Prandtlm>    moist Prandtl number [default: 1]
    --F=<F>                  basic state buoyancy difference [default: 0]
    --alpha=<alpha>          Clausius Clapeyron parameter [default: 3.0]
    --beta=<beta>            beta parameter [default: 1.201]
    --gamma=<gamma>          condensational heating parameter [default: 0.293]
    --DeltaT=<DeltaT>        Temperature at top [default: -1.0]
    --sigma2=<sigma2>        Initial condition sigma2 [default: 0.05]
    --q0=<q0>                Initial condition q0 [default: 5.]
    --nx=<nx>                x (Fourier) resolution [default: 256]
    --ny=<ny>                y (Fourier) resolution [default: 256]
    --nz=<nz>                vertical z (Chebyshev) resolution [default: 64]
    --Lx=<Lx>                x length  [default: 10.]
    --Ly=<Ly>                y length [default: 10.]
    --Lz=<Lz>                vertical z length [default: 1.]
    --restart=<restart_file> Restart from checkpoint
    --nondim=<nondim>        nondimensionalization (buoyancy or RB) [default: buoyancy]
    --filter=<filter>        fraction of modes to keep in ICs [default: 0.5]
    --mesh=<mesh>            processor mesh (you're in charge of making this consistent with nproc) [default: None]
"""
from docopt import docopt
import os
import sys
import numpy as np
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

args = docopt(__doc__)
# Parameters
Lx = float(args['--Lx'])
Ly = float(args['--Ly'])
Lz = float(args['--Lz'])
nx = int(args['--nx'])
ny = int(args['--ny'])
nz = int(args['--nz'])
mesh = args['--mesh']
if mesh == 'None':
    mesh = None
else:
    mesh = [int(i) for i in mesh.split(',')]
if ny == 0:
    threeD = False
    Ly = 0. # override Ly if 2D
else:
    threeD = True

betaval = float(args['--beta'])
Rayleigh = float(args['--Rayleigh'])    # The real Rayleigh number is this number times a buoyancy difference
Prandtl = float(args['--Prandtl'])
Prandtlm = float(args['--Prandtlm'])
Fval = float(args['--F'])
alphaval = float(args['--alpha'])
gammaval = float(args['--gamma'])
DeltaTval = float(args['--DeltaT'])

# initial conditions
#sigma2 = 0.005
sigma2 = float(args['--sigma2'])
q0_amplitude = float(args['--q0'])

# Nondimensionalization
nondim = args['--nondim']

if nondim == 'RB':
    #  RB diffusive scaling
    Pval = 1                      #  diffusion on buoyancy. Always = 1 in this scaling.
    Sval = 1                      #  diffusion on moisture  k_q / k_b
    PdRval = Prandtl              #  diffusion on momentum
    PtRval = Prandtl * Rayleigh   #  Prandtl times Rayleigh = buoyancy force
    Rey = Rayleigh**(0.5)         #  scaling for timestep
    tauval   = 0.125*3./Rey       #  condensation time scale

    slices_dt = 0.025
    snap_dt = 1.0
    prof_dt = 0.01
elif nondim == 'buoyancy':                                           #  Buoyancy scaling
    Pval = (Rayleigh * Prandtl)**(-1/2)         #  diffusion on buoyancy
    Sval = (Rayleigh * Prandtlm)**(-1/2)        #  diffusion on moisture
    PdRval = (Prandtl / Rayleigh)**(1/2)        #  diffusion on momentum
    PtRval = 1                                  #  buoyancy force  = 1 always
    Rey = 1                                     #  scaling for timestep
    tauval   = 0.125*3./Rey                     #  condensation timescale

    slices_dt = 0.025*(Rayleigh* Prandtl)**(1/2)
    snap_dt = 1.0*(Rayleigh* Prandtl)**(1/2)
    prof_dt = 0.01*(Rayleigh* Prandtl)**(1/2)
    logger.info("Output timescales (in sim time): slices = {}, snapshots = {}, profiles ={}".format(slices_dt, snap_dt, prof_dt))
else:
    raise ValueError("Nondimensionalization {} not supported.".format(nondim))

# Create bases and domain
bases = []
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
bases.append(x_basis)
if threeD:
    y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
    bases.append(y_basis)
z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
bases.append(z_basis)
domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)
# 3D Boussinesq hydrodynamics
variables = ['p','b','u','w','bz','uz','wz','temp','q','qz']
if threeD:
    variables += ['v', 'vz']

problem = de.IVP(domain,
                 variables=variables)

# save data in directory named after script
data_dir = "scratch/" + sys.argv[0].split('.py')[0]
data_dir +="_Ra{0:5.02e}_beta{1:5.02e}_Pr{2:5.02e}_Prm{3:5.02e}_F{4:5.02e}_alpha{5:5.02e}_gamma{6:5.02e}_DeltaT{7:5.02e}_sigma2{8:5.02e}_q0{9:5.02e}_nondim:{10:s}_nx{11:d}_ny{12:d}_nz{13:d}_Lx{14:5.02e}_Ly{15:5.02e}_Lz{16:5.02e}".format(Rayleigh, betaval, Prandtl, Prandtlm, Fval, alphaval, gammaval, DeltaTval, sigma2, q0_amplitude, nondim, nx, ny, nz, Lx, Ly, Lz)
logger.info("saving run in: {}".format(data_dir))

if domain.distributor.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.makedirs('{:s}/'.format(data_dir))

problem.parameters['P'] = Pval
problem.parameters['PdR'] = PdRval
problem.parameters['PtR'] = PtRval
problem.parameters['M'] = 0.19
problem.parameters['S'] = 1.0
problem.parameters['beta'] = betaval
problem.parameters['tau'] = tauval
problem.parameters['alpha'] = alphaval
problem.parameters['DeltaT'] = DeltaTval

# numerics parameters
problem.parameters['k'] = 1e5 # cutoff for tanh
problem.parameters['Lx'] = Lx
if threeD:
    problem.parameters['Ly'] = Ly

if threeD:
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
    problem.substitutions['KE'] = '0.5*(u*u + v*v + w*w)'
else:
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
    problem.substitutions['KE'] = '0.5*(u*u + w*w)'

problem.substitutions['H(A)'] = '0.5*(1. + tanh(k*A))'
problem.substitutions['qs'] = 'exp(alpha*temp)'
problem.substitutions['rh'] = 'q/exp(alpha*temp)'

if threeD:
    problem.add_equation('dx(u) + dy(v) + wz = 0')

    problem.add_equation('dt(b) - P*(dx(dx(b)) + dy(dy(b)) + dz(bz)) = - u*dx(b) - v*dy(b) - w*bz + M*H(q - qs)*(q - qs)/tau')
    problem.add_equation('dt(q) - S*(dx(dx(q)) + dy(dy(q)) + dz(qz)) = - u*dx(q) - v*dy(q) - w*qz +   H(q - qs)*(qs - q)/tau')

    problem.add_equation('dt(u) - PdR*(dx(dx(u)) + dy(dy(u)) + dz(uz)) + dx(p)                = - u*dx(u) - v*dy(u) - w*uz')
    problem.add_equation('dt(v) - PdR*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + dy(p)                = - u*dx(v) - v*dy(v) - w*vz')
    problem.add_equation('dt(w) - PdR*(dx(dx(w)) + dy(dy(w)) + dz(wz)) + dz(p) - PtR*b = - u*dx(w) - v*dy(w) - w*wz')
else:
    problem.add_equation('dx(u) + wz = 0')

    problem.add_equation('dt(b) - P*(dx(dx(b)) + dz(bz)) = - u*dx(b) - w*bz + M*H(q - qs)*(q - qs)/tau')
    problem.add_equation('dt(q) - S*(dx(dx(q)) + dz(qz)) = - u*dx(q) - w*qz +   H(q - qs)*(qs - q)/tau')

    problem.add_equation('dt(u) - PdR*(dx(dx(u)) + dz(uz)) + dx(p)                = - u*dx(u) - w*uz')
    problem.add_equation('dt(w) - PdR*(dx(dx(w)) + dz(wz)) + dz(p) - PtR*b        = - u*dx(w) - w*wz')
    

problem.add_equation('bz - dz(b) = 0')
problem.add_equation('qz - dz(q) = 0')
problem.add_equation('uz - dz(u) = 0')
if threeD:
    problem.add_equation('vz - dz(v) = 0')
problem.add_equation('wz - dz(w) = 0')
problem.add_equation('dz(temp) - bz = -beta')

problem.add_bc('left(b) = 0')
problem.add_bc('right(b) = beta + DeltaT')
problem.add_bc('left(q) = 1')
problem.add_bc('right(q) = exp(alpha*DeltaT)')
problem.add_bc('left(u) = 0')
problem.add_bc('right(u) = 0')
if threeD:
    problem.add_bc('left(v) = 0')
    problem.add_bc('right(v) = 0')
    
problem.add_bc('left(w) = 0')
problem.add_bc('left(temp) = 0')

if threeD:
    cond1 = 'nx != 0 or ny != 0'
    cond2 = 'nx == 0 and ny == 0'
else:
    cond1 = 'nx != 0'
    cond2 = 'nx == 0'
problem.add_bc('right(w) = 0', condition=cond1)
problem.add_bc('right(p) = 0', condition=cond2)

# Build solver
ts = de.timesteppers.SBDF3
solver = problem.build_solver(ts)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
if threeD:
    y = domain.grid(1)
    z = domain.grid(2)
else:
    z = domain.grid(1)
b = solver.state['b']
bz = solver.state['bz']
q = solver.state['q']
qz = solver.state['qz']

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval

gshape = problem.domain.dist.grid_layout.global_shape(scales=problem.domain.dealias)
slices = problem.domain.dist.grid_layout.slices(scales=problem.domain.dealias)
rand = np.random.RandomState(seed=42)
pert = rand.standard_normal(gshape)[slices]

#b['g'] = -0.0*(z - pert)
b['g'] = 0#T1ovDTval-(1.00-betaval)*z
b.differentiate('z', out=bz)
#q['g'] = q0val*np.exp(-betaval*z/T0val)+1e-2*np.exp(-((x-1.0)/0.01)^2)*np.exp(-((z-0.5)/0.01)^2)
#q['g'] = q0val*np.exp(-betaval*z/T0val)+(1e-2)*np.exp(-((z-0.5)*(z-0.5)/0.02))*np.exp(-((x-1.0)*(x-1.0)/0.02))

q['g'] = q0_amplitude*np.exp(-((z-0.1)*(z-0.1)/sigma2))*np.exp(-((x-1.0)*(x-1.0)/sigma2))
if threeD:
    q['g'] *= np.exp(-((y-1.0)*(y-1.0)/sigma2))
q.differentiate('z', out=qz)

# Integration parameters
dt = 1e-4
solver.stop_sim_time = 3000
solver.stop_wall_time = 3600. * 24. * 4.9
solver.stop_iteration = np.inf

hermitian_cadence = 100

# CFL routines
logger.info("Starting CFL")
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=5, safety=0.3,
                     max_change=1.5, min_change=0.5)
cfl_vels = ['u','w']
if threeD:
    cfl_vels.append('v')
CFL.add_velocities(cfl_vels)

# Analysis
slices = solver.evaluator.add_file_handler(os.path.join(data_dir, 'slices'), sim_dt=slices_dt, max_writes=50)
if threeD:
    slices.add_task('interp(b, z = 0.5)', name='b midplane')
    slices.add_task('interp(u, z = 0.5)', name='u midplane')
    slices.add_task('interp(v, z = 0.5)', name='v midplane')
    slices.add_task('interp(w, z = 0.5)', name='w midplane')
    slices.add_task('interp(temp, z = 0.5)', name='temp midplane')
    slices.add_task('interp(q, z = 0.5)', name='q midplane')
    slices.add_task('interp(rh, z = 0.5)', name='rh midplane')

    slices.add_task('interp(b, x = 0)', name='b vertical')
    slices.add_task('interp(u, x = 0)', name='u vertical')
    slices.add_task('interp(v, x = 0)', name='v vertical')
    slices.add_task('interp(w, x = 0)', name='w vertical')
    slices.add_task('interp(temp, x = 0)', name='temp vertical')
    slices.add_task('interp(q, x = 0)', name='q vertical')
    slices.add_task('interp(rh, x = 0)', name='rh vertical')
else:
    slices.add_task('b', name='b vertical')
    slices.add_task('u', name='u vertical')
    slices.add_task('w', name='w vertical')
    slices.add_task('temp', name='temp vertical')
    slices.add_task('q', name='q vertical')
    slices.add_task('rh', name='rh vertical')

snapshots = solver.evaluator.add_file_handler(os.path.join(data_dir, 'snapshots'), sim_dt=snap_dt, max_writes=10)
snapshots.add_system(solver.state)

profiles = solver.evaluator.add_file_handler(os.path.join(data_dir, 'profiles'), sim_dt=prof_dt)
profiles.add_task('plane_avg(b)', name='b')
profiles.add_task('plane_avg(u)', name='u')
if threeD:
    profiles.add_task('plane_avg(v)', name='v')
profiles.add_task('plane_avg(w)', name='w')
profiles.add_task('plane_avg(q)', name='q')
profiles.add_task('plane_avg(temp)', name='temp')

# Main loop
dt = CFL.compute_dt()

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("KE", name='KE')

try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration - 1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e, max E_kin: %e' %(solver.iteration, solver.sim_time, dt, flow.max('KE')))

        if (solver.iteration - 1) % hermitian_cadence == 0:
            for field in solver.state.fields:
                field.require_grid_space()
        dt = CFL.compute_dt()

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    # Print statistics
    logger.info('Run time: %f' %(end_time-start_time))
    logger.info('Iterations: %i' %solver.iteration)

