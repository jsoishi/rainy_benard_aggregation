import os
import numpy as np
import dedalus.public as de
import h5py
import matplotlib.pyplot as plt
import logging
from etools import Eigenproblem

logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
    logging.getLogger(system).setLevel(logging.WARNING)


import analytic_atmosphere
from analytic_zc import f_zc as zc_analytic
from analytic_zc import f_Tc as Tc_analytic

class RainyBenardEVP():
    def __init__(self, nz, Ra, tau_in, kx_in, γ, α, β, lower_q0, k, atmosphere=None, relaxation_method=None, Legendre=True, erf=True, nondim='buoyancy', bc_type=None, Prandtl=1, Prandtlm=1, Lz=1, dealias=3/2, dtype=np.complex128, twoD=True, label=None):
        logger.info('Ra = {:}, kx = {:}, α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(Ra,kx_in,α,β,γ,tau_in, k))
        self.nz = nz
        self.Lz = Lz

        self.dealias = dealias
        self.α = α
        self.β = β
        self.γ = γ
        self.lower_q0 = lower_q0
        self.k = k
        self.atmosphere = atmosphere

        self.Prandtl = Prandtl
        self.Prandtlm = Prandtlm

        self.twoD = twoD
        if self.twoD:
            self.coords = de.CartesianCoordinates('x', 'z')
        else:
            self.coords = de.CartesianCoordinates('x', 'y', 'z')
        self.dist = de.Distributor(self.coords, dtype=dtype)
        self.erf = erf
        self.Legendre = Legendre
        self.nondim = nondim
        self.bc_type = bc_type
        if self.Legendre:
            self.zb = de.Legendre(self.coords.coords[-1], size=self.nz, bounds=(0, self.Lz), dealias=self.dealias)
        else:
            self.zb = de.ChebyshevT(self.coords.coords[-1], size=self.nz, bounds=(0, self.Lz), dealias=self.dealias)

        self.kx = self.dist.Field(name='kx')
        self.kx['g'] = kx_in
        # protection against array type-casting via scipy.optimize;
        # important when updating Lx during that loop.
        # kx = np.float64(kx_in).squeeze()[()]

        self.Rayleigh = self.dist.Field(name='Ra')
        self.Rayleigh['g'] = Ra
        self.tau = self.dist.Field(name='tau')
        self.tau['g'] = tau_in
        self.relaxation_method = relaxation_method

        if self.atmosphere:
            self.load_atmosphere()
        else:
            self.build_atmosphere(label=label)
        self.build_solver()

    def build_atmosphere(self, label=None):
        logger.info("Building atmosphere")
        if self.lower_q0 < 1:
            atm_name = 'unsaturated'
        elif self.lower_q0 == 1:
            atm_name = 'saturated'
        else:
            raise ValueError("lower q0 has invalid value, q0 = {:}".format(self.lower_q0))

        self.case_name = 'analytic_{:s}/alpha{:1.0f}_beta{:}_gamma{:}_q{:1.1f}'.format(atm_name, self.α,self.β,self.γ, self.lower_q0)

        self.case_name += '/tau{:}_k{:.3e}_relaxation_{:}'.format(self.tau['g'].squeeze().real,self.k,self.relaxation_method)
        if self.erf:
            self.case_name += '_erf'
        if self.Legendre:
            self.case_name += '_Legendre'
        if label:
            self.case_name += '_'+label

        if atm_name == 'unsaturated':
            zc = zc_analytic()(self.γ)
            Tc = Tc_analytic()(self.γ)
            sol = analytic_atmosphere.unsaturated(self.dist, self.zb, self.β, self.γ, zc, Tc, dealias=self.dealias, q0=self.lower_q0, α=self.α)
        elif atm_name == 'saturated':
            sol = analytic_atmosphere.saturated(self.dist, self.zb, self.β, self.γ, dealias=self.dealias, q0=self.lower_q0, α=self.α)

        self.b0 = sol['b']
        self.b0.name = 'b0'
        self.q0 = sol['q']
        self.q0.name = 'q0'
        # use only gradient in z direction.
        self.grad_b0 = de.grad(self.b0).evaluate()
        self.grad_q0 = de.grad(self.q0).evaluate()

        if not os.path.exists('{:s}/'.format(self.case_name)) and self.dist.comm.rank == 0:
            os.makedirs('{:s}/'.format(self.case_name))

    def plot_background(self,label=None):
        fig, ax = plt.subplots(ncols=2, figsize=[12,6])
        qs0 = self.qs0.evaluate()
        qs0.change_scales(1)
        self.b0.change_scales(1)
        self.q0.change_scales(1)
        z = self.zb.local_grid(1)
        zd = self.zb.local_grid(self.dealias)
        p0 = ax[0].plot(self.b0['g'].squeeze().real, z.squeeze(), label=r'$b$')
        p1 = ax[0].plot(self.γ*self.q0['g'].squeeze().real, z.squeeze(), label=r'$\gamma q$')
        p2 = ax[0].plot(self.b0['g'].squeeze().real+self.γ*self.q0['g'].squeeze().real, z.squeeze(), label=r'$m = b + \gamma q$')
        p3 = ax[0].plot(self.γ*qs0['g'].squeeze().real, z.squeeze(), linestyle='dashed', alpha=0.3, label=r'$\gamma q_s$')
        ax2 = ax[0].twiny()
        self.scrN.change_scales(1)
        p4 = ax2.plot(self.scrN['g'].squeeze().real, z.squeeze(), color='xkcd:purple grey', label=r'$\mathcal{N}(z)$')
        ax2.set_xlabel(r'$\mathcal{N}(z)$')
        ax2.xaxis.label.set_color('xkcd:purple grey')
        ax2.set_xlim(-0.1,1.1)
        lines = p0 + p1 + p2 + p3 + p4
        labels = [l.get_label() for l in lines]
        ax[0].legend(lines, labels)
        ax[0].set_xlabel(r'$b$, $\gamma q$, $m$')
        ax[0].set_ylabel(r'$z$')
        #ax[1].plot(q0['g'][0,0,:]-qs0['g'][0,0,:], z[0,0,:])
        ax[1].plot(de.grad(self.b0).evaluate()['g'][-1].squeeze().real, zd.squeeze(), label=r'$\nabla b$')
        ax[1].plot(de.grad(self.γ*self.q0).evaluate()['g'][-1].squeeze().real, zd.squeeze(), label=r'$\gamma \nabla q$')
        ax[1].plot(de.grad(self.b0+self.γ*self.q0).evaluate()['g'][-1].squeeze().real, zd.squeeze(), label=r'$\nabla m$')
        ax[1].set_xlabel(r'$\nabla b$, $\gamma \nabla q$, $\nabla m$')
        ax[1].legend()
        ax[1].axvline(x=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5)
        fig.tight_layout()
        tau_val = self.tau["g"].squeeze().real
        Ra_val  = self.Rayleigh["g"].squeeze().real
        filebase = self.case_name+f'/nz_{self.nz}_k_{self.k}_tau_{tau_val:0.1e}_Ra_{Ra_val:0.2e}_evp_background'
        if label:
            filebase += f'_{label}'
        fig.savefig(filebase+'.png', dpi=300)


    def build_solver(self):
        if self.twoD:
            ex, ez = self.coords.unit_vector_fields(self.dist)
            ey = self.dist.VectorField(self.coords)
            ey['c'] = 0
        else:
            ex, ey, ez = self.coords.unit_vector_fields(self.dist)
        dx = lambda A: 1j*kx*A # 1-d mode onset
        dy = lambda A: 0*A # flexibility to add 2-d mode if desired

        grad = lambda A: de.grad(A) + ex*dx(A) + ey*dy(A)
        div = lambda A:  de.div(A) + dx(ex@A) + dy(ey@A)
        lap = lambda A: de.lap(A) + dx(dx(A)) + dy(dy(A))

        trans = lambda A: de.TransposeComponents(A)

        z = self.zb.local_grid(1)
        zd = self.zb.local_grid(self.dealias)

        zbasis = self.zb
        bases = (self.zb)
        bases_p = ()

        p = self.dist.Field(name='p', bases=bases)
        u = self.dist.VectorField(self.coords, name='u', bases=bases)
        b = self.dist.Field(name='b', bases=bases)
        q = self.dist.Field(name='q', bases=bases)
        τp = self.dist.Field(name='τp')
        τu1 = self.dist.VectorField(self.coords, name='τu1') #, bases=bases_p)
        τu2 = self.dist.VectorField(self.coords, name='τu2') #, bases=bases_p)
        τb1 = self.dist.Field(name='τb1') #, bases=bases_p)
        τb2 = self.dist.Field(name='τb2') #, bases=bases_p)
        τq1 = self.dist.Field(name='τq1') #, bases=bases_p)
        τq2 = self.dist.Field(name='τq2') #, bases=bases_p)
        variables = [p, u, b, q, τp, τu1, τu2, τb1, τb2, τq1, τq2]
        varnames = [v.name for v in variables]
        self.fields = {k:v for k, v in zip(varnames, variables)}

        lift_basis = zbasis.derivative_basis(1)
        lift = lambda A: de.Lift(A, lift_basis, -1)

        lift_basis2 = zbasis.derivative_basis(2)
        lift2 = lambda A: de.Lift(A, lift_basis2, -1)
        lift2_2 = lambda A: de.Lift(A, lift_basis2, -2)

        z_grid = self.dist.Field(name='z_grid', bases=self.zb)
        z_grid['g'] = z

        # need local aliases...this is a weakness of this approach
        Lz = self.Lz
        kx = self.kx
        q0 = self.q0
        b0 = self.b0
        grad_q0 = self.grad_q0
        grad_b0 = self.grad_b0
        γ = self.γ
        α = self.α
        β = self.β
        tau = self.tau

        T0 = self.b0 - β*z_grid
        qs0 = np.exp(α*T0)#.evaluate()
        self.qs0 = qs0
        e = grad(u) + trans(grad(u))

        from scipy.special import erf
        if self.nondim == 'diffusion':
            P = 1                      #  diffusion on buoyancy. Always = 1 in this scaling.
            S = self.Prandtlm               #  diffusion on moisture  k_q / k_b
            PdR = self.Prandtl              #  diffusion on momentum
            PtR = self.Prandtl*self.Rayleigh     #  Prandtl times Rayleigh = buoyancy force
        elif self.nondim == 'buoyancy':
            P = (self.Rayleigh * self.Prandtl)**(-1/2)         #  diffusion on buoyancy
            S = (self.Rayleigh * self.Prandtlm)**(-1/2)        #  diffusion on moisture
            PdR = (self.Rayleigh/self.Prandtl)**(-1/2)         #  diffusion on momentum
            PtR = 1
            #tau_in /=                     # think through what this should be
        else:
            raise ValueError('nondim {:} not in valid set [diffusion, buoyancy]'.format(nondim))

        if self.erf:
            H = lambda A: 0.5*(1+erf(self.k*A))
        else:
            H = lambda A: 0.5*(1+np.tanh(self.k*A))
        # solve NLBVP for smoothing background
        b0_lower = b0(z=0).evaluate()['g'].squeeze()[()]
        b0_upper = b0(z=Lz).evaluate()['g'].squeeze()[()]
        q0_lower = q0(z=0).evaluate()['g'].squeeze()[()]
        q0_upper = q0(z=Lz).evaluate()['g'].squeeze()[()]

        τb01 = self.dist.Field(name='τb01')
        τb02 = self.dist.Field(name='τb02')
        τq01 = self.dist.Field(name='τq01')
        τq02 = self.dist.Field(name='τq02')
        vars0 = [q0, b0, τb01, τb02, τq01, τq02]
        lap0 = lambda A: de.lap(A)
        logger.info("relaxing atmosphere via {:}".format(self.relaxation_method))
        for system in ['subsystems']:
            logging.getLogger(system).setLevel(logging.WARNING)
        if self.relaxation_method == 'NLBVP':
            nlbvp = de.NLBVP(vars0, namespace=locals())
            nlbvp.add_equation('-P*lap0(b0) + lift2(τb01) + lift2_2(τb02) = γ/tau*(q0-qs0)*H(q0-qs0)')
            nlbvp.add_equation('-S*lap0(q0) + lift2(τq01) + lift2_2(τq02) = -1/tau*(q0-qs0)*H(q0-qs0)')
            nlbvp.add_equation('b0(z=0) = b0_lower')
            nlbvp.add_equation('b0(z=Lz) = b0_upper')
            nlbvp.add_equation('q0(z=0) = q0_lower')
            nlbvp.add_equation('q0(z=Lz) = q0_upper')
            nlbvp_solver = nlbvp.build_solver()
            pert_norm = np.inf
            tol = 1e-5
            while pert_norm > tol:
                nlbvp_solver.newton_iteration(damping=0.95)
                pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in nlbvp_solver.perturbations)
                logger.info("L2 err = {:.1g}".format(pert_norm))
        elif self.relaxation_method == 'IVP':
            ivp = de.IVP(vars0, namespace=locals())
            ivp.add_equation('dt(b0) - P*lap0(b0) + lift2(τb01) + lift2_2(τb02) = γ/tau*(q0-qs0)*H(q0-qs0)')
            ivp.add_equation('dt(q0) - S*lap0(q0) + lift2(τq01) + lift2_2(τq02) = -1/tau*(q0-qs0)*H(q0-qs0)')
            ivp.add_equation('b0(z=0) = b0_lower')
            ivp.add_equation('b0(z=Lz) = b0_upper')
            ivp.add_equation('q0(z=0) = q0_lower')
            ivp.add_equation('q0(z=Lz) = q0_upper')
            ivp_solver = ivp.build_solver(de.SBDF2)
            Δt = tau['g'].squeeze().real/4
            end_time = (1/P).evaluate()['g'].squeeze().real
            while ivp_solver.sim_time < end_time:
                ivp_solver.step(Δt)
            logger.info(f"evolved atmosphere to t={ivp_solver.sim_time:.2g} using Δt={Δt} and {ivp_solver.iteration} steps")
        if self.erf:
            scrN = (H(q0 - qs0) + 1/2*(q0 - qs0)*self.k*2*(np.pi)**(-1/2)*np.exp(-self.k**2*(q0 - qs0)**2)).evaluate()
        else:
            scrN = (H(q0 - qs0) + 1/2*(q0 - qs0)*self.k*(1-(np.tanh(self.k*(q0 - qs0)))**2)).evaluate()
        scrN.name='scrN'
        self.scrN = scrN

        grad_b0 = de.grad(b0).evaluate()
        grad_q0 = de.grad(q0).evaluate()

        ω = self.dist.Field(name='ω')
        dt = lambda A: ω*A

        self.problem = de.EVP(variables, eigenvalue=ω, namespace=locals())
        self.problem.add_equation('div(u) + lift(τp) = 0')
        self.problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b*ez + lift2(τu1) + lift2_2(τu2) = 0')
        self.problem.add_equation('dt(b) - P*lap(b) + u@grad_b0 - γ/tau*(q-α*qs0*b)*scrN + lift2(τb1) + lift2_2(τb2) = 0')
        self.problem.add_equation('dt(q) - S*lap(q) + u@grad_q0 + 1/tau*(q-α*qs0*b)*scrN + lift2(τq1) + lift2_2(τq2) = 0')
        self.problem.add_equation('b(z=0) = 0')
        self.problem.add_equation('b(z=Lz) = 0')
        self.problem.add_equation('q(z=0) = 0')
        self.problem.add_equation('q(z=Lz) = 0')
        if self.bc_type=='stress-free':
            logger.info("BCs: bottom stress-free")
            self.problem.add_equation('ez@u(z=0) = 0')
            self.problem.add_equation('ez@(ex@e(z=0)) = 0')
            if not self.twoD:
                self.problem.add_equation('ez@(ey@e(z=0)) = 0')
        else:
            logger.info("BCs: bottom no-slip")
            self.problem.add_equation('u(z=0) = 0')
            #self.problem.add_equation('ez@u(z=0) = 0')
            #self.problem.add_equation("ez@τu1 = 0")
        if self.bc_type == 'top-stress-free' or self.bc_type == 'stress-free':
            logger.info("BCs: top stress-free")
            self.problem.add_equation('ez@u(z=Lz) = 0')
            self.problem.add_equation('ez@(ex@e(z=Lz)) = 0')
            if not self.twoD:
                self.problem.add_equation('ez@(ey@e(z=Lz)) = 0')
        else:
            logger.info("BCs: top no-slip")
            self.problem.add_equation('u(z=Lz) = 0')
        self.problem.add_equation('integ(p) = 0')
        self.solver = self.problem.build_solver(ncc_cutoff=1e-10)

    def solve(self, dense=True, N_evals=20, target=0):
        if dense:
            self.solver.solve_dense(self.solver.subproblems[0], rebuild_matrices=True)
            self.solver.eigenvalues = self.solver.eigenvalues[np.isfinite(self.solver.eigenvalues)]
        else:
            self.solver.solve_sparse(self.solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
            #self.solver.print_subproblem_ranks(target=target)
        self.eigenvalues = self.solver.eigenvalues

def mode_reject(lo_res, hi_res, drift_threshold=1e6, plot_drift_ratios=True):
    ep = Eigenproblem(None,use_ordinal=False, drift_threshold=drift_threshold)
    ep.evalues_low   = lo_res.eigenvalues
    ep.evalues_high  = hi_res.eigenvalues
    evals_good, indx = ep.discard_spurious_eigenvalues()

    if plot_drift_ratios:
        fig, ax = plt.subplots()
        ep.plot_drift_ratios(axes=ax)
        nz = lo_res.nz
        fig.savefig(f'{lo_res.case_name}/nz_{nz}_drift_ratios.png', dpi=300)

    return evals_good, indx, ep
