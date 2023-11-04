import os
import numpy as np
from mpi4py import MPI
from scipy.special import lambertw as W
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

class SplitRainyBenardEVP():
    def __init__(self, nz, Ra, tau_in, kx_in, γ, α, β, lower_q0, k, Legendre=True, erf=True, nondim='buoyancy', bc_type=None, Prandtl=1, Prandtlm=1, Lz=1, dealias=3/2, dtype=np.complex128, twoD=True):
        logger.info('Ra = {:}, kx = {:}, α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(Ra,kx_in,α,β,γ,tau_in, k))
        self.nz = nz
        self.Lz = Lz

        self.dealias = dealias
        self.α = α
        self.β = β
        self.γ = γ
        self.lower_q0 = lower_q0
        self.k = k

        self.Prandtl = Prandtl
        self.Prandtlm = Prandtlm

        self.twoD = twoD
        if self.twoD:
            self.coords = de.CartesianCoordinates('x', 'z')
            self.vector_dims = 3
            self.z_slice = (0,slice(None))
        else:
            self.coords = de.CartesianCoordinates('x', 'y', 'z')
            self.vector_dims = 4
            self.z_slice = (0,0,slice(None))
        self.dist = de.Distributor(self.coords, dtype=dtype, comm=MPI.COMM_SELF)
        self.erf = erf
        self.Legendre = Legendre
        self.nondim = nondim
        self.bc_type = bc_type

        self.get_zc_Tc()
        if self.Legendre:
            self.zb1 = de.Legendre(self.coords['z'], size=self.nz, bounds=(0, self.zc), dealias=self.dealias)
            self.zb2 = de.Legendre(self.coords['z'], size=self.nz, bounds=(self.zc, self.Lz), dealias=self.dealias)
        else:
            self.zb1 = de.ChebyshevT(self.coords['z'], size=self.nz, bounds=(0, self.zc), dealias=self.dealias)
            self.zb2 = de.ChebyshevT(self.coords['z'], size=self.nz, bounds=(self.zc, self.Lz), dealias=self.dealias)
        self.z = np.concatenate([self.zb1.local_grid(1).squeeze(), self.zb2.local_grid(1).squeeze()])
        self.zd = np.concatenate([self.zb1.local_grid(self.dealias).squeeze(), self.zb2.local_grid(self.dealias).squeeze()])

        # self.kx = self.dist.Field(name='kx')
        # self.kx['g'] = kx_in
        # # protection against array type-casting via scipy.optimize;
        # # important when updating Lx during that loop.
        kx = np.float64(kx_in).squeeze()[()]
        # # Build Fourier basis for x with prescribed kx as the fundamental mode
        self.nx = 4
        self.Lx = 2 * np.pi / kx
        self.xb = de.ComplexFourier(self.coords['x'], size=self.nx, bounds=(0, self.Lx), dealias=self.dealias)

        self.Rayleigh = self.dist.Field(name='Ra')
        self.Rayleigh['g'] = Ra
        self.tau = self.dist.Field(name='tau')
        self.tau['g'] = tau_in
        self.build_atmosphere()
        self.build_solver()
    
    def get_zc_Tc(self):
        #if self.α != 3.0 or self.β != 1.2 or self.lower_q0 != 0.6:
        #    raise NotImplementedError("Only α = 3.0, β = 1.2, and lower_q0 = 0.6 currently supported.")
        if self.α != 3.0 or self.lower_q0 !=0.6:
            raise NotImplementedError("Only α = 3.0 and lower_q0 = 0.6 currently supported.")
        if self.γ == 0.3:
            self.zc = 0.483289354408442
            self.Tc = -0.4588071140209613
        elif self.γ == 0.19:
            self.zc = 0.4751621541611023
            self.Tc = -0.4588071140209616
        else:
            raise NotImplementedError(f"gamma = {self.gamma:.3f} is not supported.")

    def build_atmosphere(self):
        logger.info("Building atmosphere")
        atm_name = 'unsaturated'

        self.case_name = 'analytic_{:s}/stacked_alpha{:1.0f}_beta{:}_gamma{:}_q{:1.1f}'.format(atm_name, self.α,self.β,self.γ, self.lower_q0)
        self.case_name += '/tau{:}_k{:.3e}'.format(self.tau['g'].squeeze().real,self.k)
        if self.erf:
            self.case_name += '_erf'
        if self.Legendre:
            self.case_name += '_Legendre'
        z1 = self.zb1.local_grid(1)
        z2 = self.zb2.local_grid(1)
        ΔT = -1
        b1 = 0
        b2 = self.β + ΔT
        q1 = self.lower_q0
        q2 = np.exp(self.α*ΔT)
        
        bc = self.Tc + self.β*self.zc
        qc = np.exp(self.α*self.Tc)

        P = bc + self.γ*qc
        Q = ((b2-bc) + self.γ*(q2-qc))
        C = P + Q*(z2-self.zc)/(1-self.zc) - self.β*z2
        T_lo = self.Tc*z1/self.zc
        T_hi = C - W(self.α*self.γ*np.exp(self.α*C)).real/self.α
        
        b0_lo = self.dist.Field(name='b0_lo', bases=self.zb1)
        b0_lo['g'] = T_lo + self.β*z1
        b0_hi = self.dist.Field(name='b0_hi', bases=self.zb2)
        b0_hi['g'] = T_hi + self.β*z2
        q0_lo = self.dist.Field(name='q0_lo', bases=self.zb1)
        q0_lo['g'] = q1 + (qc - q1)*z1/self.zc
        q0_hi = self.dist.Field(name='q0_hi', bases=self.zb2)
        q0_hi['g'] = np.exp(self.α*T_hi)
        qs0_lo = self.dist.Field(name='qs0_lo', bases=self.zb1)
        qs0_lo['g'] = np.exp(self.α*T_lo)
        
        self.b0 = [b0_lo,b0_hi]
        self.q0 = [q0_lo,q0_hi]
        self.qs0 = [qs0_lo, q0_hi] # above zc, qs0 = q0
        self.grad_b0 = []
        self.grad_q0 = []
        for b0,q0 in zip(self.b0, self.q0):
            self.grad_b0.append(de.grad(b0).evaluate())
            self.grad_q0.append(de.grad(q0).evaluate())
        if not os.path.exists('{:s}/'.format(self.case_name)) and self.dist.comm.rank == 0:
            os.makedirs('{:s}/'.format(self.case_name))

    def concatenate_bases(self, field1, field2):
        return np.concatenate([field1['g'],field2['g']], axis=-1)
        
    def plot_background(self,label=None):
        fig, ax = plt.subplots(ncols=2, figsize=[12,6])
        for b0,q0 in zip(self.b0, self.q0):
            b0.change_scales(1)
            q0.change_scales(1)
        b0 = self.concatenate_bases(*self.b0)
        q0 = self.concatenate_bases(*self.q0)
        qs0 = self.concatenate_bases(*self.qs0)
        grad_q0 = self.concatenate_bases(*self.grad_q0)
        grad_b0 = self.concatenate_bases(*self.grad_b0)

        p0 = ax[0].plot(b0[0,:].real, self.z, label=r'$b$')
        p1 = ax[0].plot(self.γ*q0[0,:].real, self.z, label=r'$\gamma q$')
        p2 = ax[0].plot(b0[0,:].real+self.γ*q0[0,:].real, self.z, label=r'$m = b + \gamma q$')
        p3 = ax[0].plot(self.γ*qs0[0,:].real, self.z, linestyle='dashed', alpha=0.3, label=r'$\gamma q_s$')
        lines = p0 + p1 + p2 + p3 
        labels = [l.get_label() for l in lines]
        ax[0].legend(lines, labels)
        ax[0].set_xlabel(r'$b$, $\gamma q$, $m$')
        ax[0].set_ylabel(r'$z$')
        
        ax[1].plot(grad_b0[1,0,:].real, self.zd, label=r'$\nabla b$')
        ax[1].plot(self.γ*grad_q0[1,0,:].real, self.zd, label=r'$\gamma \nabla q$')
        ax[1].plot(grad_b0[1,0,:].real+self.γ*grad_q0[1,0,:].real, self.zd, label=r'$\nabla m$')
        ax[1].set_xlabel(r'$\nabla b$, $\gamma \nabla q$, $\nabla m$')
        ax[1].legend()
        ax[1].axvline(x=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5)
        for a in ax:
            a.axhline(self.zc,color='k',alpha=0.4)
        fig.tight_layout()
        tau_val = self.tau["g"].squeeze().real
        Ra_val  = self.Rayleigh["g"].squeeze().real
        filebase = self.case_name+f'/nz_{self.nz}_k_{self.k}_tau_{tau_val:0.1e}_Ra_{Ra_val:0.2e}_evp_background_stacked'
        if label:
            filebase += f'_{label}'
        fig.savefig(filebase+'.png', dpi=300)

    def plot_eigenmode(self, index, mode_label=None):
        self.solver.set_state(index,0)
        fields = ['b','q','p','u']
        names = {}
        data ={}
        for f in fields:
            lower_field = self.fields[f'{f}1']
            lower_field.change_scales(1)
            upper_field = self.fields[f'{f}2']
            upper_field.change_scales(1)
            data[f] = self.concatenate_bases(lower_field,upper_field)
            names[f] = lower_field.name

        fig, axes = plt.subplot_mosaic([['ux','bzoom','uz'],
                                        ['b', 'q','p']], layout='constrained')
        i_max = np.argmax(np.abs(data['b'][self.z_slice]))
        phase_correction = data['b'][self.z_slice][i_max]

        for v in ['q','p','b']:
            name = names[v]
            d = data[v]/phase_correction
            axes[v].plot(d[0,:].real, self.z)            
            axes[v].plot(d[0,:].imag, self.z, ':')       
            axes[v].set_xlabel(f"${name}$")              
            axes[v].set_ylabel(r"$z$")                   
            axes[v].axhline(self.zc, color='k',alpha=0.3)

        axes['bzoom'].plot(d[0,:].real, self.z,'x-')            
        axes['bzoom'].plot(d[0,:].imag, self.z, ':')
        axes['bzoom'].set_ylim(0.47,0.4775)
        axes['bzoom'].set_xlabel(f"${name}$")              
        axes['bzoom'].set_ylabel(r"$z$")                   
        axes['bzoom'].axhline(self.zc, color='k',alpha=0.3)

        u = data['u']/phase_correction
        axes['ux'].plot(u[0,0,...,:].squeeze().real, self.z)
        axes['ux'].plot(u[0,0,...,:].squeeze().imag, self.z,':')
        axes['ux'].set_xlabel(r"$u_x$")
        axes['ux'].set_ylabel(r"$z$")
        axes['uz'].plot(u[-1,0,...,:].squeeze().real, self.z)
        axes['uz'].plot(u[-1,0,...,:].squeeze().imag, self.z, ':')
        axes['uz'].set_xlabel(r"$u_z$")
        axes['uz'].set_ylabel(r"$z$")
        axes['q'].set_title(f"phase {phase_correction.real:.3e}+{phase_correction.imag:.3e}i")
        sigma = self.solver.eigenvalues[index]
        fig.suptitle(f"$\sigma = {sigma.real:.2f} {sigma.imag:+.2e} i$")
        if not mode_label:
            mode_label = index
        kx = 2*np.pi/self.Lx
        fig_filename=f"emode_indx_{mode_label}_Ra_{self.Rayleigh['g'].squeeze().real:.2e}_nz_{self.nz}_kx_{kx:.3f}_bc_{self.bc_type}"
        fig.savefig(self.case_name +'/'+fig_filename+'.pdf')
        logger.info("eigenmode {:d} saved in {:s}".format(index, self.case_name +'/'+fig_filename+'.png'))


    def build_solver(self):
        if self.twoD:
            ex, ez = self.coords.unit_vector_fields(self.dist)
            ey = self.dist.VectorField(self.coords)
            ey['c'] = 0
        else:
            raise NotImplementedError("3D is not implemented.")
        grad = lambda A: de.grad(A) 
        div = lambda A:  de.div(A) 
        lap = lambda A: de.lap(A) 
        trans = lambda A: de.TransposeComponents(A)
        
        z1 = self.zb1.local_grid(1)
        z2 = self.zb2.local_grid(1)

        bases1 = (self.xb, self.zb1)
        bases2 = (self.xb, self.zb2)
        bases_p = self.xb
        p1 = self.dist.Field(name='p1', bases=bases1)
        u1 = self.dist.VectorField(self.coords, name='u1', bases=bases1)
        b1 = self.dist.Field(name='b1', bases=bases1)
        q1 = self.dist.Field(name='q1', bases=bases1)
        τp = self.dist.Field(name='τp')
        τu11 = self.dist.VectorField(self.coords, name='τu11', bases=bases_p)
        τu21 = self.dist.VectorField(self.coords, name='τu21', bases=bases_p)
        τb11 = self.dist.Field(name='τb11', bases=bases_p)
        τb21 = self.dist.Field(name='τb21', bases=bases_p)
        τq11 = self.dist.Field(name='τq11', bases=bases_p)
        τq21 = self.dist.Field(name='τq21', bases=bases_p)

        p2 = self.dist.Field(name='p2', bases=bases2)
        u2 = self.dist.VectorField(self.coords, name='u2', bases=bases2)
        b2 = self.dist.Field(name='b2', bases=bases2)
        q2 = self.dist.Field(name='q2', bases=bases2)
        τu12 = self.dist.VectorField(self.coords, name='τu12', bases=bases_p)
        τu22 = self.dist.VectorField(self.coords, name='τu22', bases=bases_p)
        τb12 = self.dist.Field(name='τb12', bases=bases_p)
        τb22 = self.dist.Field(name='τb22', bases=bases_p)
        τq12 = self.dist.Field(name='τq12', bases=bases_p)
        τq22 = self.dist.Field(name='τq22', bases=bases_p)
        variables = [p1, u1, b1, q1, τp, τu11, τu21, τb11, τb21, τq11, τq21,
                     p2, u2, b2, q2, τu12, τu22, τb12, τb22, τq12, τq22]
        varnames = [v.name for v in variables]
        self.fields = {k:v for k, v in zip(varnames, variables)}

        lift_basis1 = self.zb1
        lift1 = lambda A, n: de.Lift(A, lift_basis1, n)
        lift_basis2 = self.zb2
        lift2 = lambda A, n: de.Lift(A, lift_basis2, n)

        # need local aliases...this is a weakness of this approach
        Lz = self.Lz
        #kx = self.kx
        γ = self.γ
        α = self.α
        β = self.β
        tau = self.tau
        zc = self.zc

        grad_q01 = self.grad_q0[0]
        grad_b01 = self.grad_b0[0]
        qs02 = self.qs0[1]
        grad_q02 = self.grad_q0[1]
        grad_b02 = self.grad_b0[1]
        e1 = grad(u1) + trans(grad(u1))
        e2 = grad(u2) + trans(grad(u2))
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

        ω = self.dist.Field(name='ω')
        dt = lambda A: ω*A

        self.problem = de.EVP(variables, eigenvalue=ω, namespace=locals())
        for i in [1, 2]:
            self.problem.add_equation(f'div(u{i}) + τp + 1/PdR*dot(lift{i}(τu2{i},-1),ez) = 0')
            self.problem.add_equation(f'dt(u{i}) - PdR*lap(u{i}) + grad(p{i}) - PtR*b{i}*ez + lift{i}(τu1{i}, -1) + lift{i}(τu2{i}, -2) = 0')
        # self.problem.add_equation('div(u1) + τp1 + 1/PdR*dot(lift1(τu21,-1),ez) = 0')
        # self.problem.add_equation('dt(u1) - PdR*lap(u1) + grad(p1) - PtR*b1*ez + lift1(τu11, -1) + lift1(τu21, -2) = 0')
        # self.problem.add_equation('div(u2) + τp2 + 1/PdR*dot(lift2(τu22,-1),ez) = 0')
        # self.problem.add_equation('dt(u2) - PdR*lap(u2) + grad(p2) - PtR*b2*ez + lift2(τu12, -1) + lift2(τu22, -2) = 0')
        # unsaturated layer
        self.problem.add_equation('dt(b1) - P*lap(b1) + u1@grad_b01 + lift1(τb11, -1) + lift1(τb21, -2) = 0')
        self.problem.add_equation('dt(q1) - S*lap(q1) + u1@grad_q01 + lift1(τq11, -1) + lift1(τq21, -2) = 0')
        # saturated layer
        self.problem.add_equation('dt(b2) - P*lap(b2) + u2@grad_b02 - γ/tau*(q2-α*qs02*b2) + lift2(τb12, -1) + lift2(τb22, -2) = 0')
        self.problem.add_equation('dt(q2) - S*lap(q2) + u2@grad_q02 + 1/tau*(q2-α*qs02*b2) + lift2(τq12, -1) + lift2(τq22, -2) = 0')

        # matching conditions
        self.problem.add_equation('p1(z=zc) - p2(z=zc) = 0')
        self.problem.add_equation('b1(z=zc) - b2(z=zc) = 0')
        self.problem.add_equation('q1(z=zc) - q2(z=zc) = 0')
        self.problem.add_equation('u1(z=zc) - u2(z=zc) = 0')
        self.problem.add_equation('ez@grad(b1)(z=zc) - ez@grad(b2)(z=zc) = 0')
        self.problem.add_equation('ez@grad(q1)(z=zc) - ez@grad(q2)(z=zc) = 0')
        self.problem.add_equation('ez@grad(ex@u1)(z=zc) - ez@grad(ex@u2)(z=zc) = 0')
        # boundary conditions
        self.problem.add_equation('b1(z=0) = 0')
        self.problem.add_equation('b2(z=Lz) = 0')
        self.problem.add_equation('q1(z=0) = 0')
        self.problem.add_equation('q2(z=Lz) = 0')
        if self.bc_type=='stress-free':
            logger.info("BCs: bottom stress-free")
            self.problem.add_equation('ez@u1(z=0) = 0')
            self.problem.add_equation('ez@(ex@e1(z=0)) = 0')
            if not self.twoD:
                self.problem.add_equation('ez@(ey@e1(z=0)) = 0')
        else:
            logger.info("BCs: bottom no-slip")
            self.problem.add_equation('u1(z=0) = 0')
        if self.bc_type == 'top-stress-free' or self.bc_type == 'stress-free':
            logger.info("BCs: top stress-free")
            self.problem.add_equation('ez@u2(z=Lz) = 0')
            self.problem.add_equation('ez@(ex@e2(z=Lz)) = 0')
            if not self.twoD:
                self.problem.add_equation('ez@(ey@e2(z=Lz)) = 0')
        else:
            logger.info("BCs: top no-slip")
            self.problem.add_equation('u2(z=Lz) = 0')
        self.problem.add_equation('integ(p1) + integ(p2) = 0')
        self.solver = self.problem.build_solver(entry_cutoff=0)#)ncc_cutoff=1e-10)

    def solve(self, dense=True, N_evals=20, target=0):
        if dense:
            self.solver.solve_dense(self.solver.subproblems[1], rebuild_matrices=True)
        else:
            self.solver.solve_sparse(self.solver.subproblems[1], N=N_evals, target=target, rebuild_matrices=True)
        self.eigenvalues = self.solver.eigenvalues

class RainyBenardEVP():
    def __init__(self, nz, Ra, tau_in, kx_in, γ, α, β, lower_q0, k, atmosphere=None, relaxation_method=None, Legendre=True, erf=True, nondim='buoyancy', bc_type=None, Prandtl=1, Prandtlm=1, Lz=1, dealias=3/2, dtype=np.complex128, twoD=True):
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
            self.z_slice = (0,slice(None))
        else:
            self.coords = de.CartesianCoordinates('x', 'y', 'z')
            self.z_slice = (0,0,slice(None))
        self.dist = de.Distributor(self.coords, dtype=dtype)
        self.erf = erf
        self.Legendre = Legendre
        self.nondim = nondim
        self.bc_type = bc_type
        if self.Legendre:
            self.zb = de.Legendre(self.coords.coords[-1], size=self.nz, bounds=(0, self.Lz), dealias=self.dealias)
        else:
            self.zb = de.ChebyshevT(self.coords.coords[-1], size=self.nz, bounds=(0, self.Lz), dealias=self.dealias)
            self.z = self.zb.local_grid(1).squeeze()
        # protection against array type-casting via scipy.optimize;
        # important when updating Lx during that loop.
        kx = np.float64(kx_in).squeeze()[()]
        self.kx = kx
        # Build Fourier basis for x with prescribed kx as the fundamental mode
        self.nx = 4
        self.Lx = 2 * np.pi / kx
        self.xb = de.ComplexFourier(self.coords['x'], size=self.nx, bounds=(0, self.Lx))

        self.Rayleigh = self.dist.Field(name='Ra')
        self.Rayleigh['g'] = Ra
        self.tau = self.dist.Field(name='tau')
        self.tau['g'] = tau_in
        self.relaxation_method = relaxation_method

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

        self.case_name = 'analytic_{:s}/alpha{:1.0f}_beta{:}_gamma{:}_q{:1.1f}'.format(atm_name, self.α,self.β,self.γ, self.lower_q0)

        self.case_name += '/tau{:}_k{:.3e}_relaxation_{:}'.format(self.tau['g'].squeeze().real,self.k,self.relaxation_method)
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

    def plot_eigenmode(self, index, mode_label=None):
        self.solver.set_state(index,0)
        fields = ['b','q','p','u']
        names = {}
        data ={}
        for f in fields:
            data[f] = self.fields[f]['g']
            names[f] = self.fields[f].name

        fig, axes = plt.subplot_mosaic([['ux','.','uz'],
                                        ['b', 'q','p']], layout='constrained')
        i_max = np.argmax(np.abs(data['b'][self.z_slice]))
        phase_correction = data['b'][self.z_slice][i_max]

        for v in ['b','q','p']:
            name = names[v]
            d = data[v]/phase_correction
            axes[v].plot(d[0,:].real, self.z)
            axes[v].plot(d[0,:].imag, self.z, ':')
            axes[v].set_xlabel(f"${name}$")
            axes[v].set_ylabel(r"$z$")

        u = data['u']/phase_correction
        axes['ux'].plot(u[0,0,...,:].squeeze().real, self.z)
        axes['ux'].plot(u[0,0,...,:].squeeze().imag, self.z,':')
        axes['ux'].set_xlabel(r"$u_x$")
        axes['ux'].set_ylabel(r"$z$")
        axes['uz'].plot(u[-1,0,...,:].squeeze().real, self.z)
        axes['uz'].plot(u[-1,0,...,:].squeeze().imag, self.z, ':')
        axes['uz'].set_xlabel(r"$u_z$")
        axes['uz'].set_ylabel(r"$z$")
        axes['q'].set_title(f"phase {phase_correction.real:.3e}+{phase_correction.imag:.3e}i")
        sigma = self.solver.eigenvalues[index]
        fig.suptitle(f"$\sigma = {sigma.real:.2f} {sigma.imag:+.2e} i$")
        if not mode_label:
            mode_label = index
        kx = 2*np.pi/self.Lx
        fig_filename=f"emode_indx_{mode_label}_Ra_{self.Rayleigh['g'].squeeze().real:.2e}_nz_{self.nz}_kx_{kx:.3f}_bc_{self.bc_type}"
        fig.savefig(self.case_name +'/'+fig_filename+'.pdf')
        logger.info("eigenmode {:d} saved in {:s}".format(index, self.case_name +'/'+fig_filename+'.png'))


    def build_solver(self):
        if self.twoD:
            ex, ez = self.coords.unit_vector_fields(self.dist)
            ey = self.dist.VectorField(self.coords)
            ey['c'] = 0
        else:
            ex, ey, ez = self.coords.unit_vector_fields(self.dist)
        #dx = lambda A: 1j*kx*A # 1-d mode onset
        dy = lambda A: 0*A # flexibility to add 2-d mode if desired

        grad = lambda A: de.grad(A) #+ ex*dx(A) + ey*dy(A)
        div = lambda A:  de.div(A) #+ dx(ex@A) + dy(ey@A)
        lap = lambda A: de.lap(A) # + dx(dx(A)) + dy(dy(A))

        trans = lambda A: de.TransposeComponents(A)

        z = self.zb.local_grid(1)
        zd = self.zb.local_grid(self.dealias)

        bases = (self.xb, self.zb)
        bases_p = (self.xb)

        p = self.dist.Field(name='p', bases=bases)
        u = self.dist.VectorField(self.coords, name='u', bases=bases)
        b = self.dist.Field(name='b', bases=bases)
        q = self.dist.Field(name='q', bases=bases)
        τp = self.dist.Field(name='τp')
        τu1 = self.dist.VectorField(self.coords, name='τu1', bases=bases_p)
        τu2 = self.dist.VectorField(self.coords, name='τu2', bases=bases_p)
        τb1 = self.dist.Field(name='τb1', bases=bases_p)
        τb2 = self.dist.Field(name='τb2', bases=bases_p)
        τq1 = self.dist.Field(name='τq1', bases=bases_p)
        τq2 = self.dist.Field(name='τq2', bases=bases_p)
        variables = [p, u, b, q, τp, τu1, τu2, τb1, τb2, τq1, τq2]
        varnames = [v.name for v in variables]
        self.fields = {k:v for k, v in zip(varnames, variables)}

        lift_basis = self.zb #.derivative_basis(2)
        lift = lambda A, n: de.Lift(A, lift_basis, n)

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
            nlbvp.add_equation('-P*lap0(b0) + lift(τb01, -1) + lift(τb02, -2) = γ/tau*(q0-qs0)*H(q0-qs0)')
            nlbvp.add_equation('-S*lap0(q0) + lift(τq01, -1) + lift(τq02, -2) = -1/tau*(q0-qs0)*H(q0-qs0)')
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
            ivp.add_equation('dt(b0) - P*lap0(b0) + lift(τb01, -1) + lift(τb02, -2) = γ/tau*(q0-qs0)*H(q0-qs0)')
            ivp.add_equation('dt(q0) - S*lap0(q0) + lift(τq01, -1) + lift(τq02, -2) = -1/tau*(q0-qs0)*H(q0-qs0)')
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

        grad_b0 = grad(b0).evaluate()
        grad_q0 = grad(q0).evaluate()

        ω = self.dist.Field(name='ω')
        dt = lambda A: ω*A

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
            logger.info("BCs: bottom stress-free")
            self.problem.add_equation('ez@u(z=0) = 0')
            self.problem.add_equation('ez@(ex@e(z=0)) = 0')
            if not self.twoD:
                self.problem.add_equation('ez@(ey@e(z=0)) = 0')
        else:
            logger.info("BCs: bottom no-slip")
            self.problem.add_equation('u(z=0) = 0')
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
            self.solver.solve_dense(self.solver.subproblems[1], rebuild_matrices=True)
        else:
            self.solver.solve_sparse(self.solver.subproblems[1], N=N_evals, target=target, rebuild_matrices=True)
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
