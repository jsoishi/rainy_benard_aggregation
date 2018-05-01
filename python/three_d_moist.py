"""
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


"""

import os
import numpy as np
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Ly, Lz = (10., 10., 1.)
nx = 256
ny = 256
nz = 64

betaval = 1.201
Rayleigh = 1.e6    # The real Rayleigh number is this number times a buoyancy difference
Prandtl = 1
Prandtlm = 1   # moisture Prandtl number
Fval = 0       # this sets the basic state buoyancy difference, which I set to zero.
alphaval = 3.0                                  # Clausius Clapeyron parameter
gammaval = 0.293 * 1                            # condensational heating parameter
DeltaTval = -1.0                                # Temperature at top

# initial conditions
#sigma2 = 0.005
sigma2 = 0.05
q0_amplitude = 5.

# Nondimensionalization
nondim = 'buoyancy'

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
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=[16,16])
# 3D Boussinesq hydrodynamics
problem = de.IVP(domain,
                 variables=['p','b','u','v','w','bz','uz','vz','wz','temp','q','qz'])

# physics parameters
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
problem.parameters['Ly'] = Ly

problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
problem.substitutions['H(A)'] = '0.5*(1. + tanh(k*A))'
problem.substitutions['qs'] = 'exp(alpha*temp)'
problem.substitutions['rh'] = 'q/exp(alpha*temp)'

problem.add_equation('dx(u) + dy(v) + wz = 0')

problem.add_equation('dt(b) - P*(dx(dx(b)) + dy(dy(b)) + dz(bz)) = - u*dx(b) - v*dy(b) - w*bz + M*H(q - qs)*(q - qs)/tau')
problem.add_equation('dt(q) - S*(dx(dx(q)) + dy(dy(q)) + dz(qz)) = - u*dx(q) - v*dy(q) - w*qz +   H(q - qs)*(qs - q)/tau')

problem.add_equation('dt(u) - PdR*(dx(dx(u)) + dy(dy(u)) + dz(uz)) + dx(p)                = - u*dx(u) - v*dy(u) - w*uz')
problem.add_equation('dt(v) - PdR*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + dy(p)                = - u*dx(v) - v*dy(v) - w*vz')
problem.add_equation('dt(w) - PdR*(dx(dx(w)) + dy(dy(w)) + dz(wz)) + dz(p) - PtR*b = - u*dx(w) - v*dy(w) - w*wz')

problem.add_equation('bz - dz(b) = 0')
problem.add_equation('qz - dz(q) = 0')
problem.add_equation('uz - dz(u) = 0')
problem.add_equation('vz - dz(v) = 0')
problem.add_equation('wz - dz(w) = 0')
problem.add_equation('dz(temp) - bz = -beta')

problem.add_bc('left(b) = 0')
problem.add_bc('left(q) = 1')
problem.add_bc('left(u) = 0')
problem.add_bc('left(v) = 0')
problem.add_bc('left(w) = 0')
problem.add_bc('left(temp) = 0')
problem.add_bc('right(b) = beta + DeltaT')
problem.add_bc('right(q) = exp(alpha*DeltaT)')
problem.add_bc('right(u) = 0')
problem.add_bc('right(v) = 0')
problem.add_bc('right(w) = 0', condition='nx != 0 or ny != 0')
problem.add_bc('right(p) = 0', condition='nx == 0 and ny == 0')

# Build solver
ts = de.timesteppers.SBDF3
solver = problem.build_solver(ts)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)
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

q['g'] = q0_amplitude*np.exp(-((z-0.1)*(z-0.1)/sigma2))*np.exp(-((x-1.0)*(x-1.0)/sigma2))*np.exp(-((y-1.0)*(y-1.0)/sigma2))
q.differentiate('z', out=qz)

# Integration parameters
dt = 1e-4
solver.stop_sim_time = 1500
solver.stop_wall_time = 3600. * 24. * 4.9
solver.stop_iteration = np.inf

hermitian_cadence = 100

# CFL routines
logger.info("Starting CFL")
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=5, safety=0.3,
                     max_change=1.5, min_change=0.5)
CFL.add_velocities(('u', 'v', 'w'))

# Analysis
slices = solver.evaluator.add_file_handler('slices', sim_dt=slices_dt, max_writes=50)
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

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=snap_dt, max_writes=10)
snapshots.add_system(solver.state)

profiles = solver.evaluator.add_file_handler('profiles', sim_dt=prof_dt)
profiles.add_task('plane_avg(b)', name='b')
profiles.add_task('plane_avg(u)', name='u')
profiles.add_task('plane_avg(v)', name='v')
profiles.add_task('plane_avg(w)', name='w')
profiles.add_task('plane_avg(q)', name='q')
profiles.add_task('plane_avg(temp)', name='temp')

# Main loop
dt = CFL.compute_dt()

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("(u*u + v*v + w*w)", name='KE')

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

