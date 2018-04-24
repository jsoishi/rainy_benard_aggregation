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
from mpi4py import MPI
import time

from dedalus import public as de

import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Ly, Lz = (20., 20., 1.)
nx = 64
ny = 64
nz = 32

q0val= 0.000
T1ovDTval = 5.5
betaval =1.201



# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=[2,2])
# 3D Boussinesq hydrodynamics
problem = de.IVP(domain,
                 variables=['p','b','u','v','w','bz','uz','vz','wz','temp','q','qz'])

problem.parameters['Eu'] = 1.0
problem.parameters['Prandtl'] = 1.0
problem.parameters['Ra'] = 20000000.0
problem.parameters['M'] = 10.0
problem.parameters['S'] = 1.0
problem.parameters['beta']=betaval
problem.parameters['K2'] = 4e-10
problem.parameters['tau'] = 0.00005
problem.parameters['aDT'] = 3.00
#problem.parameters['aDT'] = 2.86
problem.parameters['T1ovDT'] = T1ovDTval
problem.parameters['T1'] = T1ovDTval
problem.parameters['deltaT'] = 1.00

problem.add_equation('dx(u) + dy(v) + wz = 0')

problem.add_equation('dt(b) - (dx(dx(b)) +dy(dy(b)) + dz(bz))    = - u*dx(b) - v*dy(b) - w*bz+M*0.5*(1.0+tanh(100000.*(q-K2*exp(aDT*temp))))*(q-K2*exp(aDT*temp))/tau')
problem.add_equation('dt(q) - S*(dx(dx(q)) + dy(dy(q)) + dz(qz)) = - u*dx(q) -v*dy(q) - w*qz-0.5*(1.0+tanh(100000.*(q-K2*exp(aDT*temp))))*(q-K2*exp(aDT*temp))/tau')

problem.add_equation('dt(u) - Prandtl*(dx(dx(u)) + dy(dy(u)) + dz(uz)) + Eu*dx(p)     = - u*dx(u) - v*dy(u)- w*uz')
problem.add_equation('dt(v) - Prandtl*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + Eu*dy(p)     = - u*dx(v) - v*dy(v)- w*vz')
problem.add_equation('dt(w) - Prandtl*(dx(dx(w)) + dy(dy(w)) + dz(wz)) + Eu*dz(p) - Prandtl*Ra*b = - u*dx(w) - v*dy(w) - w*wz')

problem.add_equation('bz - dz(b) = 0')
problem.add_equation('qz - dz(q) = 0')
problem.add_equation('uz - dz(u) = 0')
problem.add_equation('vz - dz(v) = 0')
problem.add_equation('wz - dz(w) = 0')
problem.add_equation('dz(temp)-bz = -beta')

problem.add_bc('left(b) = T1ovDT')
problem.add_bc('left(q) = K2*exp(aDT*T1ovDT)')
problem.add_bc('left(u) = 0')
problem.add_bc('left(v) = 0')
problem.add_bc('left(w) = 0')
problem.add_bc('left(temp) =T1ovDT')
problem.add_bc('right(b) = T1ovDT-1.0+beta')
problem.add_bc('right(q) = K2*exp(aDT*(T1ovDT-1.0))')
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
pert =  1e-3 * np.random.standard_normal(domain.local_grid_shape) * (zt - z) * (z - zb)
#b['g'] = -0.0*(z - pert)
b['g'] = T1ovDTval-(1.00-betaval)*z
b.differentiate('z', out=bz)
#q['g'] = q0val*np.exp(-betaval*z/T0val)+1e-2*np.exp(-((x-1.0)/0.01)^2)*np.exp(-((z-0.5)/0.01)^2)
#q['g'] = q0val*np.exp(-betaval*z/T0val)+(1e-2)*np.exp(-((z-0.5)*(z-0.5)/0.02))*np.exp(-((x-1.0)*(x-1.0)/0.02))
q['g'] = (2e-2)*np.exp(-((z-0.1)*(z-0.1)/0.005))*np.exp(-((x-1.0)*(x-1.0)/0.005))
q.differentiate('z', out=qz)

# Integration parameters
dt = 1e-6
solver.stop_sim_time = 6.5
solver.stop_wall_time = 6000 * 60.
solver.stop_iteration = np.inf

hermitian_cadence = 100



# CFL routines
evaluator = solver.evaluator
evaluator.vars['grid_delta_x'] = domain.grid_spacing(0)
evaluator.vars['grid_delta_y'] = domain.grid_spacing(1)
evaluator.vars['grid_delta_z'] = domain.grid_spacing(2)

cfl_cadence = 100
cfl_variables = evaluator.add_dictionary_handler(iter=cfl_cadence)
cfl_variables.add_task('u/grid_delta_x', name='f_u')
cfl_variables.add_task('v/grid_delta_y', name='f_v')
cfl_variables.add_task('w/grid_delta_z', name='f_w')

def cfl_dt():
    if z.size > 0:
        max_f_u = np.max(np.abs(cfl_variables.fields['f_u']['g']))
        max_f_v = np.max(np.abs(cfl_variables.fields['f_v']['g']))
        max_f_w = np.max(np.abs(cfl_variables.fields['f_w']['g']))
    else:
        max_f_u = max_f_v = max_f_w = 0
    max_f = max(max_f_u, max_f_v, max_f_w)
    if max_f > 0:
        min_t = 1 / max_f
    else:
        min_t = np.inf
    return min_t

safety = 0.3
dt_array = np.zeros(1, dtype=np.float64)
def update_dt(dt):
    new_dt = max(0.5*dt, min(safety*cfl_dt(), 1.1*dt))
    if domain.distributor.size > 1:
        dt_array[0] = new_dt
        domain.distributor.comm_cart.Allreduce(MPI.IN_PLACE, dt_array, op=MPI.MIN)
        new_dt = dt_array[0]
    return new_dt

solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Ly'] = Ly

# Analysis
snapshots = evaluator.add_file_handler('slices', sim_dt=0.025, max_writes=50)
snapshots.add_task('interp(b,z = 0.5)', name='b midplane')
snapshots.add_task('interp(u,z = 0.5)', name='u midplane')
snapshots.add_task('interp(v,z = 0.5)', name='v midplane')
snapshots.add_task('interp(w,z = 0.5)', name='w midplane')
snapshots.add_task('interp(temp,z = 0.5)', name='temp midplane')
snapshots.add_task('interp(q,z = 0.5)', name='q midplane')

snapshots.add_task('interp(b,x = 0)', name='b vertical')
snapshots.add_task('interp(u,x = 0)', name='u vertical')
snapshots.add_task('interp(v,x = 0)', name='v vertical')
snapshots.add_task('interp(w,x = 0)', name='w vertical')
snapshots.add_task('interp(temp,x = 0)', name='temp vertical')
snapshots.add_task('interp(q, x = 0)', name='q vertical')

snapshots = evaluator.add_file_handler('dump', sim_dt=1.0, max_writes=10)
snapshots.add_task('p')
snapshots.add_task('b')
snapshots.add_task('u')
snapshots.add_task('w')
snapshots.add_task('v')
snapshots.add_task('temp')
snapshots.add_task('q')


snapshots = evaluator.add_file_handler('profiles', sim_dt=0.01)
snapshots.add_task('Integrate(b,dx,dy)/Lx/Ly', name='b')
snapshots.add_task('Integrate(u,dx,dy)/Lx/Ly', name='u')
snapshots.add_task('Integrate(v,dx,dy)/Lx/Ly', name='v')
snapshots.add_task('Integrate(w,dx,dy)/Lx/Ly', name='w')
snapshots.add_task('Integrate(q,dx,dy)/Lx/Ly', name='q')
snapshots.add_task('Integrate(temp,dx,dy)/Lx/Ly', name='temp')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration - 1) % cfl_cadence == 0:
            dt = update_dt(dt)
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

        if (solver.iteration - 1) % hermitian_cadence == 0:
            for field in solver.state.fields:
                field.require_grid_space()

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    # Print statistics
    logger.info('Run time: %f' %(end_time-start_time))
    logger.info('Iterations: %i' %solver.iteration)

