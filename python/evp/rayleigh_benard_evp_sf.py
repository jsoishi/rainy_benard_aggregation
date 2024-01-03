"""
Dedalus script for calculating the maximum linear growth rates in no-slip
Rayleigh-Benard convection over a range of horizontal wavenumbers. This script
demonstrates solving a 1D eigenvalue problem in a Cartesian domain. It can
be ran serially or in parallel, and produces a plot of the highest growth rate
found for each horizontal wavenumber. It should take a few seconds to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard_evp.py
"""

import numpy as np
from mpi4py import MPI
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


def plot_eigenmode(solver, fields, dist, zb, index, kx, n, ω, mode_label=None, correct_phase=True, label=None):
    solver.set_state(index, solver.subproblems[1].subsystems[0])
    fig, axes = plt.subplot_mosaic([['ux','uz'],
                                    ['b','p']], layout='constrained')

    z = dist.local_grid(zb)[0,:]
    nz = z.shape[-1]
    for v in ['b','p']:
        if correct_phase:
            if v == 'b':
                i_max = np.argmax(np.abs(fields[v]['g'][0,:]))
                phase_correction = fields[v]['g'][0,i_max]
                phase_sign = np.sign(fields[v](z=0.125).evaluate()['g'][0,0])
                if np.sign(phase_correction)*np.sign(phase_sign) < 0:
                    phase_correction *= phase_sign
        else:
            phase_correction = 1
        fields[v].change_scales(1)
        name = fields[v].name
        data = fields[v]['g'][0,:]/phase_correction
        axes[v].plot(data.real, z)
        axes[v].plot(data.imag, z, ':')
        axes[v].set_xlabel(f"${name}$")
        axes[v].set_ylabel(r"$z$")
    fields['u'].change_scales(1)
    u = fields['u']['g']/phase_correction
    axes['ux'].plot(u[0,0,:].real, z)
    axes['ux'].plot(u[0,0,:].imag, z,':')
    axes['ux'].set_xlabel(r"$u_x$")
    axes['ux'].set_ylabel(r"$z$")
    axes['uz'].plot(u[1,0,:].real, z)
    axes['uz'].plot(u[1,0,:].imag, z, ':')
    axes['uz'].set_xlabel(r"$u_z$")
    axes['uz'].set_ylabel(r"$z$")
    # analytic solution
    axes['b'].plot(np.sin(n*np.pi*z), z, zorder=0, linewidth=10, alpha=0.3)
    uz_amp = (-1j*ω+(n*np.pi)**2+kx**2)
    #uz_amp = u[-1,0,np.argmax(np.abs(u[-1,0,:]))]
    axes['uz'].plot(uz_amp*np.sin(n*np.pi*z), z, zorder=0, linewidth=10, alpha=0.3)
    ux_amp = (1j*n*np.pi/kx) * uz_amp
    print(ux_amp)
    axes['ux'].plot((ux_amp*np.cos(n*np.pi*z)).imag, z, zorder=0, linewidth=10, alpha=0.3)
    if correct_phase:
        axes['b'].set_title(r"b_m = "+"{:.3g}, {:.3g}".format(phase_correction.real, phase_correction.imag))
    sigma = solver.eigenvalues[index]
    fig.suptitle(f"$\sigma = {sigma.real:.3f} {sigma.imag:+.3e} i$, mode "+label)
    if not mode_label:
        mode_label = index
    fig_filename=f"emode_indx_{mode_label}_nz_{nz}_kx_{kx:.3f}_dedalus-examples"
    fig.savefig('.' +'/'+fig_filename+'.png', dpi=300)
    logger.info("eigenmode {:d} saved in {:s}".format(index, '.' +'/'+fig_filename+'.png'))

def max_growth_rate(Rayleigh, Prandtl, kx, Nz, NEV=10, target=0, plot_critical_eigenmode=False):
    """Compute maximum linear growth rate."""

    # Parameters
    Lz = 1
    # Build Fourier basis for x with prescribed kx as the fundamental mode
    Nx = 4
    Lx = 2 * np.pi / kx

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.complex128, comm=MPI.COMM_SELF)
    xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(0, Lx))
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz))

    # Fields
    omega = dist.Field(name='omega')
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    b2 = dist.Field(name='b2', bases=(xbasis,zbasis))

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction
    dt = lambda A: -1j*omega*A
    e = grad_u + d3.trans(grad_u)

    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.EVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals(), eigenvalue=omega)
    problem.add_equation("trace(grad_u) + tau_p = 0")
    # problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) - ez@u = 0")
    # problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = 0")
    problem.add_equation("dt(b) - div(grad_b) + lift(tau_b2) - ez@u = 0")
    problem.add_equation("dt(u) - Prandtl*div(grad_u) + grad(p) - Rayleigh*Prandtl*b*ez + lift(tau_u2) = 0")
    problem.add_equation("b(z=0) = 0")
    problem.add_equation('ez@u(z=0) = 0')
    problem.add_equation('ez@(ex@e(z=0)) = 0')
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation('ez@u(z=Lz) = 0')
    problem.add_equation('ez@(ex@e(z=Lz)) = 0')
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(entry_cutoff=0)
    solver.solve_sparse(solver.subproblems[1], NEV, target=target)
    index = np.argsort(solver.eigenvalues.imag)
    # if plot_critical_eigenmode:
    #     for i in [-1, -2, -3, -4, -5, -6]:
    #         solver.set_state(index[i], solver.subproblems[1].subsystems[0])
    #         print('pre:  index {:d}, b[c] max amp = {:}'.format(index[i], np.max(np.abs(b['c']))))(
    #         print('      index {:d}, b[g] max amp = {:}'.format(index[i], np.max(np.abs(b['g']))))
    #         print('post: index {:d}, b[c] max amp = {:}'.format(index[i], np.max(np.abs(b['c']))))

    # from sympy solutions
    ω_analytic_p = lambda n: 1j*(1*np.sqrt(Rayleigh)*kx/np.sqrt(kx**2+np.pi**2*n**2) - (kx**2+np.pi**2*n**2))
    ω_analytic_m = lambda n: 1j*(-1*np.sqrt(Rayleigh)*kx/np.sqrt(kx**2+np.pi**2*n**2) - (kx**2+np.pi**2*n**2))

    if plot_critical_eigenmode:
        variables = [p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2]
        varnames = [v.name for v in variables]
        fields = {k:v for k, v in zip(varnames, variables)}
        for j, i in enumerate([-1, -2, -3, -4, -5, -6]):
            n = int(j/2)+1
            if np.mod(i,2):
                ω_analytic = ω_analytic_p(n)
                label = rf'$\omega_p, n={n}$'
            else:
                ω_analytic = ω_analytic_m(n)
                label = rf'$\omega_m, n={n}$'

            ω_evp = solver.eigenvalues[index[i]]
            print('eigenvalue: {:}; close to analytic? {:}'.format(ω_evp, np.isclose(ω_evp,ω_analytic)))

            plot_eigenmode(solver, fields, dist, zbasis, index[i], kx, n, ω_evp, label=label)

    return np.max(solver.eigenvalues.imag)


if __name__ == "__main__":

    import time
    import matplotlib.pyplot as plt
    comm = MPI.COMM_WORLD

    # Parameters
    Nz = 64
    eps = 1e-3
    Rayleigh = 27/4*np.pi**4*(1+eps)
    Prandtl = 1
    kx_crit = np.pi/np.sqrt(2)
    NEV = 10

    # find growth rate and eigenmodes of critical mode
    growth_rate = max_growth_rate(Rayleigh, Prandtl, kx_crit, Nz, NEV=NEV, plot_critical_eigenmode=True)

    logger.info('growth rate: {:}'.format(growth_rate))
