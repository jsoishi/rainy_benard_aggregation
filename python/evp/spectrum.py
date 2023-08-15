"""
Dedalus script for plotting spectrum of static drizzle solutions to the Rainy-Benard system of equations.  

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves EVPs for an existing atmospheres, solved for by scripts in the nlbvp section.

Usage:
    convective_onset.py <case> [options]

Options:
    <case>           Case (or cases) to calculate onset for

                      Properties of analytic atmosphere, if used
    --alpha=<alpha>   alpha value [default: 3]
    --beta=<beta>     beta value  [default: 1.1]
    --gamma=<gamma>   gamma value [default: 0.19]
    --q0=<q0>         basal q value [default: 0.6]

    --tau=<tau>       If set, override value of tau
    --k=<k>           If set, override value of k

    --nondim=<n>      Non-Nondimensionalization [default: buoyancy]

    --Ra=<Ra>         Minimum Rayleigh number to sample [default: 1e4]
    --kx=<kx>         x wavenumber [default: 3.14159]
    --top-stress-free Stress-free upper boundary
    --stress-free     Stress-free both boundaries

    --nz=<nz>         Number of coeffs to use in eigenvalue search; if not set, uses resolution of background
    --target=<targ>   Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>     Target number of eigenvalues to search for [default: 20]

    --erf             Use an erf rather than a tanh for the phase transition
    --Legendre        Use Legendre polynomials

    --dense           Solve densely for all eigenvalues (slow)
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
    logging.getLogger(system).setLevel(logging.WARNING)

import os
import numpy as np
import dedalus.public as de
import h5py
import matplotlib.pyplot as plt

from etools import Eigenproblem
from docopt import docopt
args = docopt(__doc__)

N_evals = int(float(args['--eigs']))
target = float(args['--target'])

dealias = 3/2
dtype = np.complex128

Prandtlm = 1
Prandtl = 1
Rayleigh = float(args['--Ra'])
kx = float(args['--kx'])
tau_in = float(args['--tau'])

Lz = 1
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)
dealias = 2

case = args['<case>']
if case == 'analytic':
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

    case += '_{:s}/alpha{:}_beta{:}_gamma{:}_q{:}'.format(atm_name, args['--alpha'],args['--beta'],args['--gamma'], args['--q0'])

    case += '/tau{:}_k{:}'.format(args['--tau'],args['--k'])
    if args['--erf']:
        case += '_erf'

    nz = int(float(args['--nz']))
    if args['--Legendre']:
        zb = de.Legendre(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
        case += '_Legendre'
    else:
        zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)

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
else:
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
    nz_sol = sol['z'].shape[0]
    f.close()
if args['--nz']:
    nz = int(float(args['--nz']))
else:
    nz = nz_sol

if args['--tau']:
    tau_in = float(args['--tau'])
if args['--k']:
    k = float(args['--k'])
logger.info('α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(α,β,γ,tau_in, k))

def build_rbc_problem(nz, coords, dist, Ra, tau_in, kx_in, γ, α, β, sol, k, Prandtl=1, Prandtlm=1, Lz=1, dealias=3/2,plot_background=False):
    kx = dist.Field(name='kx')
    Rayleigh = dist.Field(name='Ra')
    tau = dist.Field(name='tau')
    Rayleigh['g'] = Ra
    kx['g'] = kx_in
    tau['g'] = tau_in

    ex, ey, ez = coords.unit_vector_fields(dist)
    dx = lambda A: 1j*kx*A # 1-d mode onset
    dy = lambda A: 0*A # flexibility to add 2-d mode if desired

    grad = lambda A: de.grad(A) + ex*dx(A) + ey*dy(A)
    div = lambda A:  de.div(A) + dx(ex@A) + dy(ey@A)
    lap = lambda A: de.lap(A) + dx(dx(A)) + dy(dy(A))
    trans = lambda A: de.TransposeComponents(A)
    logger.info('Ra = {:}, kx = {:}, α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(Ra,kx_in,α,β,γ,tau_in, k))
    if args['--Legendre']:
        zb = de.Legendre(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
    else:
        zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
    z = zb.local_grid(1)
    zd = zb.local_grid(dealias)

    b0 = dist.Field(name='b0', bases=zb)
    q0 = dist.Field(name='q0', bases=zb)

    scale_ratio = nz_sol/nz
    b0.change_scales(scale_ratio)
    q0.change_scales(scale_ratio)
    logger.info('rescaling b0, q0 to match background from {:} to {:} coeffs (ratio: {:})'.format(nz, nz_sol, scale_ratio))

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
    variables = [p, u, b, q, τp, τu1, τu2, τb1, τb2, τq1, τq2]

    lift = lambda A, n: de.Lift(A, zb, n)

    z_grid = dist.Field(name='z_grid', bases=zb)
    z_grid['g'] = z

    T0 = b0 - β*z_grid
    qs0 = np.exp(α*T0).evaluate()

    e = grad(u) + trans(grad(u))

    from scipy.special import erf
    if args['--erf']:
        logger.info("using erf")
        H = lambda A: 0.5*(1+erf(k*A))
        scrN = (H(q0 - qs0) + 1/2*(q0 - qs0)*k*2*(np.pi)**(-1/2)*np.exp(-k**2*(q0 - qs0)**2)).evaluate()
    else:
        logger.info("using tanh")
        H = lambda A: 0.5*(1+np.tanh(k*A))
        scrN = (H(q0 - qs0) + 1/2*(q0 - qs0)*k*(1-(np.tanh(k*(q0 - qs0)))**2)).evaluate()
    scrN.name='scrN'

    grad_b0 = grad(b0).evaluate()
    grad_q0 = grad(q0).evaluate()

    ω = dist.Field(name='ω')
    dt = lambda A: ω*A

    nondim = args['--nondim']
    if nondim == 'diffusion':
        logger.info("using diffusion nondim.")
        P = 1                      #  diffusion on buoyancy. Always = 1 in this scaling.
        S = Prandtlm               #  diffusion on moisture  k_q / k_b
        PdR = Prandtl              #  diffusion on momentum
        PtR = Prandtl*Rayleigh     #  Prandtl times Rayleigh = buoyancy force
    elif nondim == 'buoyancy':
        logger.info("using buoyancy nondim.")
        P = (Rayleigh * Prandtl)**(-1/2)         #  diffusion on buoyancy
        S = (Rayleigh * Prandtlm)**(-1/2)        #  diffusion on moisture
        PdR = (Rayleigh/Prandtl)**(-1/2)         #  diffusion on momentum
        PtR = 1
        #tau_in /=                     # think through what this should be
    else:
        raise ValueError('nondim {:} not in valid set [diffusion, buoyancy]'.format(nondim))
    problem = de.EVP(variables, eigenvalue=ω, namespace=locals())
    problem.add_equation('div(u) + τp + 1/PdR*dot(lift(τu2,-1),ez) = 0')
    problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b*ez + lift(τu1, -1) + lift(τu2, -2) = 0')
    problem.add_equation('dt(b) - P*lap(b) + u@grad_b0 - γ/tau*(q-α*qs0*b)*scrN + lift(τb1, -1) + lift(τb2, -2) = 0')
    problem.add_equation('dt(q) - S*lap(q) + u@grad_q0 + 1/tau*(q-α*qs0*b)*scrN + lift(τq1, -1) + lift(τq2, -2) = 0')
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

    # without namespace passing...gives same results
    # problem.add_equation((div(u) + τp + 1/PdR*de.dot(lift(τu2,-1),ez), 0))
    # problem.add_equation((dt(u) - P*lap(u) + grad(p) - PtR*b*ez + lift(τu1, -1) + lift(τu2, -2), 0))
    # problem.add_equation((dt(b) - P*lap(b) + u@grad_b0 - γ/tau*(q-α*qs0*b)*scrN + lift(τb1, -1) + lift(τb2, -2), 0))
    # problem.add_equation((dt(q) - S*lap(q) + u@grad_q0 + 1/tau*(q-α*qs0*b)*scrN + lift(τq1, -1) + lift(τq2, -2), 0))
    # problem.add_equation((b(z=0),0))
    # problem.add_equation((b(z=Lz), 0))
    # problem.add_equation((q(z=0), 0))
    # problem.add_equation((q(z=Lz), 0))
    # if args['--stress-free']:
    #     problem.add_equation((ez@u(z=0), 0))
    #     problem.add_equation((ez@(ex@e(z=0)), 0))
    #     problem.add_equation((ez@(ey@e(z=0)), 0))
    # else:
    #     problem.add_equation((u(z=0), 0))
    # if args['--top-stress-free'] or args['--stress-free']:
    #     problem.add_equation((ez@u(z=Lz), 0))
    #     problem.add_equation((ez@(ex@e(z=Lz)), 0))
    #     problem.add_equation((ez@(ey@e(z=Lz)), 0))
    # else:
    #     problem.add_equation((u(z=Lz),0))
    # problem.add_equation((de.integ(p), 0))
    solver = problem.build_solver()

    if plot_background:
        fig, ax = plt.subplots(ncols=2, figsize=[6,6/2])
        b0.change_scales(1)
        q0.change_scales(1)
        qs0.change_scales(1)
        p0 = ax[0].plot(b0['g'][0,0,:], z[0,0,:], label=r'$b$')
        p1 = ax[0].plot(γ*q0['g'][0,0,:], z[0,0,:], label=r'$\gamma q$')
        p2 = ax[0].plot(b0['g'][0,0,:]+γ*q0['g'][0,0,:], z[0,0,:], label=r'$m = b + \gamma q$')
        p3 = ax[0].plot(γ*qs0['g'][0,0,:], z[0,0,:], linestyle='dashed', alpha=0.3, label=r'$\gamma q_s$')
        ax2 = ax[0].twiny()
        scrN.change_scales(1)
        p4 = ax2.plot(scrN['g'][0,0,:], z[0,0,:], color='xkcd:purple grey', label=r'$\mathcal{N}(z)$')
        ax2.set_xlabel(r'$\mathcal{N}(z)$')
        ax2.xaxis.label.set_color('xkcd:purple grey')
        lines = p0 + p1 + p2 + p3 + p4
        labels = [l.get_label() for l in lines]
        ax[0].legend(lines, labels)
        ax[0].set_xlabel(r'$b$, $\gamma q$, $m$')
        ax[0].set_ylabel(r'$z$')
        #ax[1].plot(q0['g'][0,0,:]-qs0['g'][0,0,:], z[0,0,:])
        ax[1].plot(grad(b0).evaluate()['g'][-1][0,0,:], zd[0,0,:], label=r'$\nabla b$')
        ax[1].plot(grad(γ*q0).evaluate()['g'][-1][0,0,:], zd[0,0,:], label=r'$\gamma \nabla q$')
        ax[1].plot(grad(b0+γ*q0).evaluate()['g'][-1][0,0,:], zd[0,0,:], label=r'$\nabla m$')
        ax[1].set_xlabel(r'$\nabla b$, $\gamma \nabla q$, $\nabla m$')
        ax[1].legend()
        ax[1].axvline(x=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5)
        fig.tight_layout()
        fig.savefig(case+'/spectrum_evp_background.png', dpi=300)


    return solver, variables

# build solvers
lo_res_sol, lo_res_vars = build_rbc_problem(    nz, coords, dist, Rayleigh, tau, kx, γ, α, β, sol, k, dealias=dealias,Lz=1)#,plot_background=True)
hi_res_sol, hi_res_vars = build_rbc_problem(3*nz/2, coords, dist, Rayleigh, tau, kx, γ, α, β, sol, k, dealias=dealias,Lz=1,plot_background=True)

dlog = logging.getLogger('subsystems')
dlog.setLevel(logging.WARNING)
for solver in [lo_res_sol, hi_res_sol]:
     if args['--dense']:
          solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
          solver.eigenvalues = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
     else:
          solver.solve_sparse(solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
ep = Eigenproblem(None)
ep.evalues_low   = lo_res_sol.eigenvalues
ep.evalues_high  = hi_res_sol.eigenvalues
evals_good, indx = ep.discard_spurious_eigenvalues()
def plot_eigenfunctions(σ):
    i_max = np.argmax(np.abs(b['g'][0,0,:]))
    phase_correction = b['g'][0,0,i_max]
    u['g'][:] /= phase_correction
    b['g'] /= phase_correction
    q['g'] /= phase_correction
    fig, ax = plt.subplots(figsize=[6,6/1.6])
    for Q in [u, q, b]:
        if Q.tensorsig:
            for i in range(3):
                p = ax.plot(Q['g'][i][0,0,:].real, z[0,0,:], label=Q.name+r'$_'+'{:s}'.format(coords.names[i])+r'$')
                ax.plot(Q['g'][i][0,0,:].imag, z[0,0,:], linestyle='dashed', color=p[0].get_color())
        else:
            p = ax.plot(Q['g'][0,0,:].real, z[0,0,:], label=Q)
            ax.plot(Q['g'][0,0,:].imag, z[0,0,:], linestyle='dashed', color=p[0].get_color())
    ax.set_title(r'$\omega_R = ${:.3g}'.format(σ.real)+ r' $\omega_I = ${:.3g}'.format(σ.imag)+' at kx = {:.3g} and Ra = {:.3g}'.format(kx['g'][0,0,0].real, Rayleigh['g'][0,0,0].real))
    ax.legend()
    fig_filename = 'eigenfunctions_{:}_Ra{:.2g}_kx{:.2g}_nz{:d}'.format(nondim, Rayleigh['g'][0,0,0].real, kx['g'][0,0,0].real, nz)
    fig.savefig(case+'/'+fig_filename+'.png', dpi=300)


indx = np.argsort(evals_good.real)
logger.info(f"fastest growing mode: {evals_good[indx][-1]}")
logger.info(f"next fastest growing mode: {evals_good[indx][-2]}")

fig, ax = plt.subplots(figsize=[6,6/1.6])
fig_filename=f"Ra_{Rayleigh:.4e}_nz_{nz}_kx_{kx}_spectrum"
#plt.scatter(hi_res_sol.eigenvalues.real, hi_res_sol.eigenvalues.imag, marker='x', label='high res')
plt.scatter(lo_res_sol.eigenvalues.real, lo_res_sol.eigenvalues.imag, marker='x', alpha=0.4,label='low res')
plt.scatter(evals_good.real, evals_good.imag, marker='o',label='good modes')
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
plt.legend()
plt.axvline(0,alpha=0.4, color='k')
plt.xlabel(r"$\Re{\sigma}$")
plt.ylabel(r"$\Im{\sigma}$")
plt.tight_layout()
fig.savefig(case+'/'+fig_filename+'.png', dpi=300)

