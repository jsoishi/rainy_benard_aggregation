import os
import numpy as np
import dedalus.public as de
import h5py
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
    logging.getLogger(system).setLevel(logging.WARNING)


import analytic_atmosphere
from analytic_zc import f_zc as zc_analytic
from analytic_zc import f_Tc as Tc_analytic

class RainyBenardEVP():
    def __init__(self, nz, Ra, tau_in, kx_in, γ, α, β, lower_q0, k, atmosphere=None, Legendre=True, erf=True, nondim='buoyancy', bc_type=None, Prandtl=1, Prandtlm=1, Lz=1, dealias=3/2, dtype=np.complex128):
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
        
        self.coords = de.CartesianCoordinates('x', 'y', 'z')
        self.dist = de.Distributor(self.coords, dtype=dtype)
        self.erf = erf
        self.Legendre = Legendre
        self.nondim = nondim
        self.bc_type = bc_type
        if self.Legendre:
            self.zb = de.Legendre(self.coords.coords[2], size=self.nz, bounds=(0, self.Lz), dealias=self.dealias)
        else:
            self.zb = de.ChebyshevT(self.coords.coords[2], size=self.nz, bounds=(0, self.Lz), dealias=self.dealias)

        self.kx = self.dist.Field(name='kx')
        self.kx['g'] = kx_in
        self.Rayleigh = self.dist.Field(name='Ra')
        self.Rayleigh['g'] = Ra
        self.tau = self.dist.Field(name='tau')
        self.tau['g'] = tau_in
        
        if self.atmosphere:
            self.load_atmosphere()
        else:
            self.build_atmosphere()
        self.build_solver()

    def build_atmosphere(self):
        logger.info("Building atmosphere")
        if self.lower_q0 < 1:
            atm_name = 'unsaturated'
        elif self.lower_q0 == 1:
            atm_name = 'saturated'
        else:
            raise ValueError("lower q0 has invalid value, q0 = {:}".format(self.lower_q0))

        self.case_name = 'analytic_{:s}/alpha{:1.0f}_beta{:}_gamma{:}_q{:1.0f}'.format(atm_name, self.α,self.β,self.γ, self.lower_q0)

        self.case_name += '/tau{:}_k{:.3e}'.format(self.tau['g'][0,0,0].real,self.k)
        if self.erf:
            self.case_name += '_erf'
        if self.Legendre:
            self.case_name += '_Legendre'

        if atm_name == 'unsaturated':
            zc = zc_analytic()(self.γ)
            Tc = Tc_analytic()(self.γ)
            sol = analytic_atmosphere.unsaturated(self.dist, self.zb, self.β, self.γ, zc, Tc, dealias=self.dealias, q0=self.lower_q0, α=self.α)
        elif atm_name == 'saturated':
            sol = analytic_atmosphere.saturated(self.dist, self.zb, self.β, self.γ, dealias=self.dealias, q0=self.lower_q0, α=self.α)

        self.b0 = sol['b']
        self.q0 = sol['q']
        # use only gradient in z direction.
        self.grad_b0 = de.grad(self.b0).evaluate()
        self.grad_q0 = de.grad(self.q0).evaluate()

        if not os.path.exists('{:s}/'.format(self.case_name)) and self.dist.comm.rank == 0:
            os.makedirs('{:s}/'.format(self.case_name))

    def build_solver(self):
        ex, ey, ez = self.coords.unit_vector_fields(self.dist)
        dx = lambda A: 1j*kx*A # 1-d mode onset
        dy = lambda A: 0*A # flexibility to add 2-d mode if desired

        grad = lambda A: de.grad(A) + ex*dx(A) + ey*dy(A)
        div = lambda A:  de.div(A) + dx(ex@A) + dy(ey@A)
        lap = lambda A: de.lap(A) + dx(dx(A)) + dy(dy(A))
        trans = lambda A: de.TransposeComponents(A)

        z = self.zb.local_grid(1)
        zd = self.zb.local_grid(self.dealias)

        p = self.dist.Field(name='p', bases=self.zb)
        u = self.dist.VectorField(self.coords, name='u', bases=self.zb)
        b = self.dist.Field(name='b', bases=self.zb)
        q = self.dist.Field(name='q', bases=self.zb)
        τp = self.dist.Field(name='τp')
        τu1 = self.dist.VectorField(self.coords, name='τu1')
        τu2 = self.dist.VectorField(self.coords, name='τu2')
        τb1 = self.dist.Field(name='τb1')
        τb2 = self.dist.Field(name='τb2')
        τq1 = self.dist.Field(name='τq1')
        τq2 = self.dist.Field(name='τq2')
        variables = [p, u, b, q, τp, τu1, τu2, τb1, τb2, τq1, τq2]
        varnames = [v.name for v in variables]
        self.fields = {k:v for k, v in zip(varnames, variables)}

        lift = lambda A, n: de.Lift(A, self.zb, n)

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
        qs0 = np.exp(α*T0).evaluate()

        e = grad(u) + trans(grad(u))

        from scipy.special import erf
        if self.erf:
            H = lambda A: 0.5*(1+erf(self.k*A))
            scrN = (H(q0 - qs0) + 1/2*(q0 - qs0)*self.k*2*(np.pi)**(-1/2)*np.exp(-self.k**2*(q0 - qs0)**2)).evaluate()
        else:
            H = lambda A: 0.5*(1+np.tanh(self.k*A))
            scrN = (H(self.q0 - qs0) + 1/2*(q0 - qs0)*k*(1-(np.tanh(k*(q0 - qs0)))**2)).evaluate()
        scrN.name='scrN'

        ω = self.dist.Field(name='ω')
        dt = lambda A: ω*A

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
        self.problem = de.EVP(variables, eigenvalue=ω, namespace=locals())
        self.problem.add_equation('div(u) + τp + 1/PdR*dot(lift(τu2,-1),ez) = 0')
        self.problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b*ez + lift(τu1, -1) + lift(τu2, -2) = 0')
        self.problem.add_equation('dt(b) - P*lap(b) + u@grad_b0 - γ/tau*(q-α*qs0*b)*scrN + lift(τb1, -1) + lift(τb2, -2) = 0')
        self.problem.add_equation('dt(q) - S*lap(q) + u@grad_q0 + 1/tau*(q-α*qs0*b)*scrN + lift(τq1, -1) + lift(τq2, -2) = 0')
        self.problem.add_equation('b(z=0) = 0')
        self.problem.add_equation('b(z=Lz) = 0')
        self.problem.add_equation('q(z=0) = 0')
        self.problem.add_equation('q(z=Lz) = 0')
        if self.bc_type=='stress-free':
            self.problem.add_equation('ez@u(z=0) = 0')
            self.problem.add_equation('ez@(ex@e(z=0)) = 0')
            self.problem.add_equation('ez@(ey@e(z=0)) = 0')
        else:
            self.problem.add_equation('u(z=0) = 0')
        if self.bc_type == 'top-stress-free' or self.bc_type == 'stress-free':
            self.problem.add_equation('ez@u(z=Lz) = 0')
            self.problem.add_equation('ez@(ex@e(z=Lz)) = 0')
            self.problem.add_equation('ez@(ey@e(z=Lz)) = 0')
        else:
            self.problem.add_equation('u(z=Lz) = 0')
        self.problem.add_equation('integ(p) = 0')
        self.solver = self.problem.build_solver()

    def solve(self, Ra, kx, dense=True, N_evals=20, target=0):
        self.kx['g'] = kx
        self.Rayleigh['g'] = Ra
        if dense:
            self.solver.solve_dense(self.solver.subproblems[0], rebuild_matrices=True)
        else:
            self.solver.solve_sparse(self.solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
        self.eigenvalues = self.solver.eigenvalues
