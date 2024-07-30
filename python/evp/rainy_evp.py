import os
import numpy as np
from mpi4py import MPI
from scipy.special import lambertw as W
from scipy.special import erf as erf_func
from mpi4py import MPI
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

class RainyEVP():
    def save(self):
        filename = f'{self.case_name}/{self.savefilename}'
        logger.info(f"saving data to {filename:}")
        with h5py.File(filename,"w") as df:
            df['eigenvalues'] = self.solver.eigenvalues
            df['eigenvectors'] = self.solver.eigenvectors

    def load(self):
        filename = f'{self.case_name}/{self.savefilename}'
        logger.info(f"loading data from {filename:}")
        sp = self.solver.subproblems[1]
        self.solver.eigenvalue_subproblem = sp
        with h5py.File(filename, "r") as df:
            self.solver.eigenvalues = df['eigenvalues'][:]
            self.solver.eigenvectors = df['eigenvectors'][:]

        self.eigenvalues = self.solver.eigenvalues

    def plot_eigenmode(self, index, mode_label=None, plot_type='png', inset=False, gamma=0.19, xlim=False, normalization='m'):
        self.solver.set_state(index,0)

        fields = ['b','q','p','u']
        names, data = self.get_names_data(fields)

        fig, axes = plt.subplot_mosaic([['ux','uz','b', 'q','m','BLANK'],], layout='constrained',figsize=(9,3),empty_sentinel="BLANK",width_ratios=[1,1,1,1,1,0.25])
        m = (data['b'] + gamma*data['q'])
        if normalization == 'm':
            logger.info("Normalizing to moist static energy.")
            norm_data = m
        else:
            logger.info("Normalizing to buoyancy.")
            norm_data = data['b']
        i_max = np.argmax(np.abs(norm_data[self.z_slice]))
        phase_correction = norm_data[self.z_slice][i_max]

        for v in ['q','b']:
            name = names[v]
            d = data[v]/phase_correction
            if self.zc:
                field_name = name[:-1]
            else:
                field_name = name
            if v == 'q':
                d*= gamma
                g = r'\gamma'
                field_name = f'${g} {field_name}$'
            else:
                field_name = f'${field_name}$'

            axes[v].plot(d[0,:].real, self.z, label=r'$\cos$')
            axes[v].plot(d[0,:].imag, self.z, ':', label=r'$\sin$')

            axes[v].set_xlabel(field_name)
            axes[v].set_ylabel(r"$z$")
            if xlim:
                axes[v].set_xlim(-1.,1.)
        m_plot = m[0,:]/phase_correction
        axes['m'].plot(m_plot.real, self.z, label=r'$\cos$')
        axes['m'].plot(m_plot.imag, self.z, ':', label=r'$\sin$')
        axes['m'].set_xlabel(f"$m$")
        axes['m'].set_ylabel(r"$z$")
        if xlim:
            axes['m'].set_xlim(-1.05,1.05)
        axes['m'].legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
        sigma = self.solver.eigenvalues[index]
        fig.text(0.83,0.05,r"$\omega$"+ f" = {sigma.real:.2f}{sigma.imag:+.2f}i", fontsize=16)
        if inset:
            b_inset_x1 = 0.99
            b_inset_y1 = 0.475
            b_inset_x2 = 1.002
            b_inset_y2 = 0.479
            axins = axes['b'].inset_axes([0.6,0.65,0.25,0.25], xlim=(b_inset_x1, b_inset_x2), ylim=(b_inset_y1, b_inset_y2), xticklabels=[], yticklabels=[])
            axins.plot(d[0,:].real, self.z, 'x-')
            if self.zc:
                axins.axhline(self.zc, color='k',alpha=0.3)
            axes['b'].indicate_inset_zoom(axins, edgecolor="black")

        # axes['bzoom'].plot(d[0,:].real, self.z,'x-')
        # axes['bzoom'].plot(d[0,:].imag, self.z, ':')
        # axes['bzoom'].set_ylim(0.47,0.4775)
        # axes['bzoom'].set_xlabel(f"${name[:-1]}$")
        # axes['bzoom'].set_ylabel(r"$z$")
        # axes['bzoom'].axhline(self.zc, color='k',alpha=0.3)

        u = data['u']/phase_correction
        axes['ux'].plot(u[0,0,...,:].squeeze().real, self.z)
        axes['ux'].plot(u[0,0,...,:].squeeze().imag, self.z,':')
        axes['ux'].set_xlabel(r"$u_x$")
        axes['ux'].set_ylabel(r"$z$")
        axes['uz'].plot(u[-1,0,...,:].squeeze().real, self.z)
        axes['uz'].plot(u[-1,0,...,:].squeeze().imag, self.z, ':')
        axes['uz'].set_xlabel(r"$u_z$")
        axes['uz'].set_ylabel(r"$z$")
        for k,v in axes.items():
            if self.zc:
                v.axhline(self.zc, color='k',alpha=0.3)
            v.set_ylim(0,1)
            if k != 'ux':
                v.get_yaxis().set_visible(False)
        logger.info(f"Phase = {phase_correction.real:.3e}+{phase_correction.imag:.3e}i")

        #fig.suptitle(f"$\sigma = {sigma.real:.2f} {sigma.imag:+.2e} i$")
        logger.info(r"$\sigma"+ f" = {sigma.real:.2f}{sigma.imag:+.2e} i$")
        if not mode_label:
            mode_label = index
        kx = 2*np.pi/self.Lx
        fig_filename=f"emode_indx_{mode_label}_Ra_{self.Rayleigh['g'].squeeze().real:.2e}_nz_{self.nz}_kx_{kx:.3f}_bc_{self.bc_type}"
        total_filename = f"{self.case_name}/{fig_filename}.{plot_type}"
        fig.savefig(total_filename)
        logger.info("eigenmode {:d} saved in {:s}".format(index, total_filename))



class SplitRainyBenardEVP(RainyEVP):
    def __init__(self, nz, Ra, tau_in, kx_in, γ, α, β, lower_q0, k, Legendre=True, erf=True, nondim='buoyancy', bc_type=None, Prandtl=1, Prandtlm=1, Lz=1, dealias=3/2, dtype=np.complex128, twoD=True, use_heaviside=False):
        self.param_string = f'Ra={Ra:}_kx={kx_in:}_α={α:}_β={β:}_γ={γ:}_tau={tau_in:}_k={k:}_nz={nz:}_bc_type={bc_type}'
        logger.info(self.param_string.replace('_',', '))
        self.savefilename = f'{self.param_string.replace("=","_"):}_eigenvectors.h5'
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
        self.use_heaviside = use_heaviside
        self.get_zc_Tc()
        if self.Legendre:
            self.zb1 = de.Legendre(self.coords['z'], size=self.nz, bounds=(0, self.zc), dealias=self.dealias)
            self.zb2 = de.Legendre(self.coords['z'], size=self.nz, bounds=(self.zc, self.Lz), dealias=self.dealias)
        else:
            self.zb1 = de.ChebyshevT(self.coords['z'], size=self.nz, bounds=(0, self.zc), dealias=self.dealias)
            self.zb2 = de.ChebyshevT(self.coords['z'], size=self.nz, bounds=(self.zc, self.Lz), dealias=self.dealias)
        self.z = np.concatenate([self.dist.local_grid(self.zb1).squeeze(), self.dist.local_grid(self.zb2).squeeze()])
        self.zd = np.concatenate([self.dist.local_grid(self.zb1, scale=self.dealias).squeeze(), self.dist.local_grid(self.zb2, scale=self.dealias).squeeze()])

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
        from analytic_zc import f_zc as zc_analytic
        from analytic_zc import f_Tc as Tc_analytic
        self.zc = zc_analytic()(self.γ)
        self.Tc = Tc_analytic()(self.γ)

    def build_atmosphere(self):
        logger.info("Building atmosphere")
        atm_name = 'unsaturated'

        self.case_name = f'analytic_{atm_name:s}/stacked_alpha{self.α:1.0f}_beta{self.β:}_gamma{self.γ:}_q{self.lower_q0:1.1f}'
        self.case_name += f'/tau{self.tau["g"].squeeze().real:}_k{self.k:.3e}'
        if self.erf:
            self.case_name += '_erf'
        if self.Legendre:
            self.case_name += '_Legendre'
        if self.use_heaviside:
            self.case_name += '_heaviside'
        z1 = self.dist.local_grid(self.zb1)
        z2 = self.dist.local_grid(self.zb2)
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

    def get_names_data(self, fields):
        names = {}
        data ={}
        for f in fields:
            lower_field = self.fields[f'{f}1']
            lower_field.change_scales(1)
            upper_field = self.fields[f'{f}2']
            upper_field.change_scales(1)
            data[f] = self.concatenate_bases(lower_field,upper_field)
            names[f] = lower_field.name
        return names, data

    def concatenate_bases(self, field1, field2):
        return np.concatenate([field1['g'],field2['g']], axis=-1)

    def plot_background(self,label=None, plot_type='png'):
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
        fig.savefig(f"{filebase}.{plot_type}", dpi=300)

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

        z1 = self.dist.local_grid(self.zb1)
        z2 = self.dist.local_grid(self.zb2)

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
        qs01 = self.qs0[0]
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

        if self.use_heaviside:
            self.scrN = []
            if self.erf:
                H = lambda A: 0.5*(1+erf_func(self.k*A))
                for i in range(2):
                    self.scrN.append((H(self.q0[i] - self.qs0[i]) + 1/2*(self.q0[i] - self.qs0[i])*self.k*2*(np.pi)**(-1/2)*np.exp(-self.k**2*(self.q0[i] - self.qs0[i])**2)).evaluate())
                    self.scrN[i].name=f'scrN{i}'
            else:
                H = lambda A: 0.5*(1+np.tanh(self.k*A))
                for i in range(2):
                    self.scrN.append((H(self.q0[i] - self.qs0[i]) + 1/2*(self.q0[i] - self.qs0[i])*self.k*(1-(np.tanh(self.k*(self.q0[i] - self.qs0[i])))**2)).evaluate())
                    self.scrN[i].name=f'scrN{i}'

            scrN1 = self.scrN[0]
            scrN2 = self.scrN[1]
        ω = self.dist.Field(name='ω')
        dt = lambda A: ω*A

        self.problem = de.EVP(variables, eigenvalue=ω, namespace=locals())
        for i in [1, 2]:
            self.problem.add_equation(f'div(u{i}) + τp + 1/PdR*dot(lift{i}(τu2{i},-1),ez) = 0')
            self.problem.add_equation(f'dt(u{i}) - PdR*lap(u{i}) + grad(p{i}) - PtR*b{i}*ez + lift{i}(τu1{i}, -1) + lift{i}(τu2{i}, -2) = 0')
        if self.use_heaviside:
            for i in [1, 2]:
                self.problem.add_equation(f'dt(b{i}) - P*lap(b{i}) + u{i}@grad_b0{i} - γ/tau*(q{i}-α*qs0{i}*b{i})*scrN{i} + lift{i}(τb1{i}, -1) + lift{i}(τb2{i}, -2) = 0')
                self.problem.add_equation(f'dt(q{i}) - S*lap(q{i}) + u{i}@grad_q0{i} + 1/tau*(q{i}-α*qs0{i}*b{i})*scrN{i} + lift{i}(τq1{i}, -1) + lift{i}(τq2{i}, -2) = 0')
        else:
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

class RainyBenardEVP(RainyEVP):
    def __init__(self, nz, Ra, tau_in, kx_in, γ, α, β, lower_q0, k, atmosphere=None, relaxation_method=None, Legendre=True, erf=True, nondim='buoyancy', bc_type=None, Prandtl=1, Prandtlm=1, Lz=1, dealias=3/2, dtype=np.complex128, twoD=True, use_heaviside=False):
        self.param_string = f'Ra={Ra:}_kx={kx_in:}_α={α:}_β={β:}_γ={γ:}_tau={tau_in:}_k={k:}_nz={nz:}'
        logger.info(self.param_string.replace('_',', '))
        self.savefilename = f'{self.param_string.replace("=","_"):}_eigenvectors.h5'
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

        self.zc = None

        self.twoD = twoD
        if self.twoD:
            self.coords = de.CartesianCoordinates('x', 'z')
            self.z_slice = (0,slice(None))
        else:
            self.coords = de.CartesianCoordinates('x', 'y', 'z')
            self.z_slice = (0,0,slice(None))
        self.dist = de.Distributor(self.coords, dtype=dtype, comm=MPI.COMM_SELF)
        self.erf = erf
        self.Legendre = Legendre
        self.nondim = nondim
        self.bc_type = bc_type
        if self.Legendre:
            self.zb = de.Legendre(self.coords.coords[-1], size=self.nz, bounds=(0, self.Lz), dealias=self.dealias)
        else:
            self.zb = de.ChebyshevT(self.coords.coords[-1], size=self.nz, bounds=(0, self.Lz), dealias=self.dealias)
        self.z = self.dist.local_grid(self.zb).squeeze()
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

        if not os.path.exists('{:s}/'.format(self.case_name)) and MPI.COMM_WORLD.rank == 0:
            os.makedirs('{:s}/'.format(self.case_name))

    def get_names_data(self, fields):
        names = {}
        data ={}
        for f in fields:
            data[f] = self.fields[f]['g']
            names[f] = self.fields[f].name

        return names, data

    def plot_background(self,label=None, plot_type='png'):
        fig, ax = plt.subplots(ncols=2, figsize=[12,6])
        qs0 = self.qs0.evaluate()
        qs0.change_scales(1)
        self.b0.change_scales(1)
        self.q0.change_scales(1)
        z = self.dist.local_grid(self.zb)
        zd = self.dist.local_grid(self.zb, scale=self.dealias)
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
        fig.savefig(f"{filebase}.{plot_type}", dpi=300)



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

        z = self.dist.local_grid(self.zb)
        zd = self.dist.local_grid(self.zb, scale=self.dealias)

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
            H = lambda A: 0.5*(1+erf_func(self.k*A))
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

def mode_reject(lo_res, hi_res, drift_threshold=1e6, plot_drift_ratios=True, plot_type='png'):
    ep = Eigenproblem(None,use_ordinal=False, drift_threshold=drift_threshold)
    ep.evalues_low   = lo_res.eigenvalues
    ep.evalues_high  = hi_res.eigenvalues
    evals_good, indx = ep.discard_spurious_eigenvalues()

    if plot_drift_ratios:
        fig, ax = plt.subplots()
        ep.plot_drift_ratios(axes=ax)
        nz = lo_res.nz
        filename=f'{lo_res.case_name}/nz_{nz}_drift_ratios.{plot_type}'
        fig.savefig(filename, dpi=300)

    return evals_good, indx, ep


class RainySpectrum():
    def __init__(self, nz, Rayleigh, tau, kx, γ, α, β, lower_q0, k, rejection_method='resolution', Legendre=True, erf=True, nondim='buoyancy', bc_type=None, Prandtl=1, Prandtlm=1, Lz=1, dealias=3/2, dtype=np.complex128, twoD=True, use_heaviside=False, quiet=True, restart=False, N_evals=5, target=0):
        self.restart = restart
        self.N_evals = N_evals
        self.target = target
        if lower_q0 == 1:
            self.EVP = RainyBenardEVP
        else:
            self.EVP = SplitRainyBenardEVP
        self.lo_res = self.EVP(nz, Rayleigh, tau, kx, γ, α, β, lower_q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside)
        if not quiet:
            self.lo_res.plot_background()
        if rejection_method == 'resolution':
            self.hi_res = self.EVP(int(2*nz), Rayleigh, tau, kx, γ, α, β, lower_q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside)
            if not quiet:
                self.hi_res.plot_background()
        elif rejection_method == 'bases':
            self.hi_res = self.EVP(nz, Rayleigh, tau, kx, γ, α, β, lower_q0, k, Legendre=not(Legendre), erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside)
            if not quiet:
                self.hi_res.plot_background(label='alternative-basis')
        else:
            raise NotImplementedError('rejection method {:s}'.format(args['--rejection_method']))

    def solve(self, dense=False, N_evals=5, target=0, quiet=False):
        for solver in [self.lo_res, self.hi_res]:
            if self.restart:
                solver.load()
            else:
                solver.solve(dense=dense, N_evals=N_evals, target=target)
                solver.save()
        evals_ok, indx_ok, ep = mode_reject(self.lo_res, self.hi_res, plot_drift_ratios=False)
        self.evals_good = evals_ok
        self.indx = indx_ok
        self.ep = ep
        if not quiet:
            logger.info(f"max growth rate = {self.evals_good[-1]}")
