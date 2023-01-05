#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
from dedalus import public as de


# In[2]:


nz = 512
tol = 1e-3
IC = 'LBVP' # 'LBVP' -> compute LBVP, 'linear' (or else) -> use linear ICs
verbose = True
q_surface = 0.6


# In[3]:


Lx = Ly = 10
Lz = 1
Rayleigh = 1e6 #1e4
Prandtl = 1
Prandtlm = 1
#tau = 1e-2 #tau_Vallis*(Rayleigh*Prandtl)**(1/2)     #  condensation timescale

α = 3
β = 1.2 #1.201
γ = 0.3 #0.19
tau_Vallis = 5e-5
ΔT = -1


# In[4]:


start_tau = 1e-3
stop_tau = 5e-5
taus = np.logspace(np.log10(start_tau), np.log10(stop_tau), num=10)
ks = np.logspace(2, 3, num=4)


# The non-dimensional tau timescale, relative to the thermal time, is:
# \begin{equation}
#     \tau = \tau_d \frac{\kappa}{H^2} = 5\times10^{-5}
# \end{equation}
# with $\tau_d$ the dimensional condensation time (Vallis et al 2019).
# 
# In buoyancy timescales,
# \begin{align}
#     \tau &= \tau_d \sqrt{\alpha \Delta T g} \\
#     &= \tau_d \sqrt{\frac{\alpha \Delta T g H^4}{\kappa^2}} \frac{\kappa}{H^2} \\
#     & = \sqrt{Ra Pr} \left(\tau_d \frac{\kappa}{H^2}\right) \\
#     & = \sqrt{Ra Pr} \times \left(5\times10^{-5}\right) 
# \end{align}
# or, given $Ra \approx 10^{6}$
# \begin{equation}
#     \tau \approx 5 \times 10^{-3}
# \end{equation}
# This indicates that, in buoyancy time units, condensation is rapid compared to buoyant times.  Maybe too rapid.
# 
# Meanwhile, the quantity $P \tau$ is:
# \begin{align}
#     P \tau &= \frac{\sqrt{Ra Pr}}{\sqrt{Ra Pr}} \times \left(5\times10^{-5}\right) \\
#     & = 5\times10^{-5}
# \end{align}
# Things probably don't get better if we multiply all terms through by P$\ldots$

# In[5]:


P = 1                                 #  diffusion on buoyancy
S = (Prandtlm/Prandtl)**(-1/2)        #  diffusion on moisture


# In[6]:


# Create bases and domain
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=np.float64)
dealias = 2
zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
z = zb.local_grid(1)

b = dist.Field(name='b', bases=zb)
q = dist.Field(name='q', bases=zb)

τb1 = dist.Field(name='τb1')
τb2 = dist.Field(name='τb2')
τq1 = dist.Field(name='τq1')
τq2 = dist.Field(name='τq2')

zb1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
zb2 = zb.clone_with(a=zb.a+2, b=zb.b+2)
lift1 = lambda A, n: de.Lift(A, zb1, n)
lift = lambda A, n: de.Lift(A, zb2, n)

ex, ey, ez = coords.unit_vector_fields(dist)

k = dist.Field(name='k')
k['g'] = 1e2 # cutoff for tanh
H = lambda A: 0.5*(1+np.tanh(k*A))

z_grid = dist.Field(name='z_grid', bases=zb)
z_grid['g'] = z

temp = b - β*z_grid
temp.name = 'T'

qs = np.exp(α*temp)
rh = q*np.exp(-α*temp)

tau = dist.Field(name='tau')


# In[7]:


def plot_solution(solution, title=None, mask=None, linestyle=None, ax=None):
    b = solution['b']
    q = solution['q']
    m = solution['m']
    T = solution['T']
    rh = solution['rh']

    for f in [b, q, m, T, rh]:
        f.change_scales(1)
        
    if mask is None:
        mask = np.ones_like(z, dtype=bool)
    if ax is None:
        fig, ax = plt.subplots(ncols=2)
        markup = True
    else:
        for axi in ax:
            axi.set_prop_cycle(None)
        markup = False
    ax[0].plot(b['g'][mask],z[mask], label='$b$', linestyle=linestyle)
    ax[0].plot(γ*q['g'][mask],z[mask], label='$\gamma q$', linestyle=linestyle)
    ax[0].plot(m['g'][mask],z[mask], label='$b+\gamma q$', linestyle=linestyle)

    ax[1].plot(T['g'][mask],z[mask], label='$T$', linestyle=linestyle)
    ax[1].plot(q['g'][mask],z[mask], label='$q$', linestyle=linestyle)
    ax[1].plot(rh['g'][mask],z[mask], label='$r_h$', linestyle=linestyle)
    
    if markup:
        ax[1].legend()
        ax[0].legend()
        ax[0].set_ylabel('z')
        if title:
            ax[0].set_title(title)
    return ax


# In[8]:


from scipy.optimize import newton
from scipy.interpolate import interp1d

def find_zc(sol, ε=1e-3, root_finding = 'inverse'):
    rh = sol['rh']
    rh.change_scales(1)
    f = interp1d(z[0,0,:], rh['g'][0,0,:])
    if root_finding == 'inverse':
        # invert the relationship and use interpolation to find where r_h = 1-ε (approach from below)
        f_i = interp1d(rh['g'][0,0,:], z[0,0,:]) #inverse
        zc = f_i(1-ε)
    elif root_finding == 'discrete':
        # crude initial emperical zc; look for where rh-1 ~ 0, in lower half of domain.
        zc = z[0,0,np.argmin(np.abs(rh['g'][0,0,0:int(nz/2)]-1))]
#    if zc is None:
#        zc = 0.2
#    zc = newton(f, 0.2)
    return zc


# In[9]:


if IC == 'LBVP':
    dt = lambda A: 0*A
    # Stable linear solution as an intial guess
    problem = de.LBVP([b, q, τb1, τb2, τq1, τq2], namespace=locals())
    problem.add_equation('dt(b) - P*lap(b) + lift(τb1, -1) + lift(τb2, -2) = 0')
    problem.add_equation('dt(q) - S*lap(q) + lift(τq1, -1) + lift(τq2, -2) = 0')
    problem.add_equation('b(z=0) = 0')
    problem.add_equation('b(z=Lz) = β + ΔT') # technically β*Lz
    problem.add_equation('q(z=0) = q_surface')
    problem.add_equation('q(z=Lz) = np.exp(α*ΔT)')
    solver = problem.build_solver()
    solver.solve()
else:
    b['g'] = (β + ΔT)*z
    q['g'] = (1-z+np.exp(α*ΔT))
    
print('b: {:.2g} -- {:.2g}'.format(b(z=0).evaluate()['g'][0,0,0], b(z=Lz).evaluate()['g'][0,0,0]))
print('q: {:.2g} -- {:.2g}'.format(q(z=0).evaluate()['g'][0,0,0], q(z=Lz).evaluate()['g'][0,0,0]))

LBVP_sol = {'b':b.copy(), 'q':q.copy(), 'm':(b+γ*q).evaluate().copy(), 'T':temp.evaluate().copy(), 'rh':rh.evaluate().copy()}
if verbose:
    plot_solution(LBVP_sol, title='LBVP solution')
if IC == 'LBVP':
    zc = find_zc(LBVP_sol)
    print('LBVP zc = {:.3}'.format(zc))
    LBVP_sol['zc'] = zc


# In[10]:


dt = lambda A: 0*A

# Stable nonlinear solution
problem = de.NLBVP([b, q, τb1, τb2, τq1, τq2], namespace=locals())
problem.add_equation('dt(b) - P*lap(b) + lift(τb1, -1) + lift(τb2, -2) = γ*H(q-qs)*(q-qs)/tau')
problem.add_equation('dt(q) - S*lap(q) + lift(τq1, -1) + lift(τq2, -2) = - H(q-qs)*(q-qs)/tau')
problem.add_equation('b(z=0) = 0')
problem.add_equation('b(z=Lz) = β + ΔT') # technically β*Lz
problem.add_equation('q(z=0) = q_surface*qs(z=0)')
problem.add_equation('q(z=Lz) = np.exp(α*ΔT)')

for system in ['subsystems']:
     logging.getLogger(system).setLevel(logging.WARNING)

NLBVP_library = {}
# Relax on tau
for tau_i in taus:
    tau['g'] = tau_i
    # Relax on k
    k_i = 1e2
    k['g'] = k_i
    solver = problem.build_solver()
    pert_norm = np.inf
    while pert_norm > tol:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info("tau = {:.1g}, k = {:.0g}, L2 err = {:.1g}".format(tau['g'][0,0,0], k['g'][0,0,0], pert_norm))

taus = [taus[-1]]

for tau_i in taus:
    tau['g'] = tau_i
    # Relax on k
    for i, k_i in enumerate(ks):
        k['g'] = k_i
        solver = problem.build_solver()
        pert_norm = np.inf
        while pert_norm > tol:
            solver.newton_iteration()
            pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
            logger.info("tau = {:.1g}, k = {:.0g}, L2 err = {:.1g}".format(tau['g'][0,0,0], k['g'][0,0,0], pert_norm))
        NLBVP_sol = {'b':b.copy(), 'q':q.copy(), 'm':(b+γ*q).evaluate().copy(), 'T':temp.evaluate().copy(), 'rh':rh.evaluate().copy()}
        zc = find_zc(NLBVP_sol)
        logger.info('tau = {:.1g}, k = {:.0g}, zc = {:.2g}'.format(tau['g'][0,0,0], k['g'][0,0,0], zc))
        NLBVP_sol['zc'] = zc
        
        # store in library of solutions
        if i == 0:
            NLBVP_library[tau_i] = {}
        NLBVP_library[tau_i][k_i] = NLBVP_sol
# In[11]:


for system in ['subsystems']:
     logging.getLogger(system).setLevel(logging.WARNING)

NLBVP_library = {}

# for first loop, use LBVP or linear solution as first guess
need_guess = False

# Relax on tau
for tau_i in taus:
    tau['g'] = tau_i
    # Relax on k
    for i, k_i in enumerate(ks):
        if need_guess:
            b.change_scales(1)
            q.change_scales(1)
            sol['b'].change_scales(1)
            sol['q'].change_scales(1)
            b['g'] = sol['b']['g']
            q['g'] = sol['q']['g']
            need_guess = False
        k['g'] = k_i
        solver = problem.build_solver()
        pert_norm = np.inf
        while pert_norm > tol:
            solver.newton_iteration()
            pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
            logger.info("tau = {:.1g}, k = {:.0g}, L2 err = {:.1g}".format(tau['g'][0,0,0], k['g'][0,0,0], pert_norm))
        NLBVP_sol = {'b':b.copy(), 'q':q.copy(),
                     'm':(b+γ*q).evaluate().copy(), 'T':temp.evaluate().copy(), 'rh':rh.evaluate().copy(), 
                     'tau':tau['g'][0,0,0], 'k':k['g'][0,0,0]}
        zc = find_zc(NLBVP_sol)
        logger.info('tau = {:.1g}, k = {:.0g}, zc = {:.2g}'.format(tau['g'][0,0,0], k['g'][0,0,0], zc))
        NLBVP_sol['zc'] = zc
        
        # store in library of solutions
        if i == 0:
            NLBVP_library[tau_i] = {}
        NLBVP_library[tau_i][k_i] = NLBVP_sol
    need_guess = True
    # use the lowest k at the current tau as the guess for the next tau; then converge on k
    sol = NLBVP_library[tau_i][ks[0]]

import json
with open('rainy_benard_NLBVP_alpha{:}_beta{:}_gamma{:}_nz{:}.json'.format(α,β,γ,nz), 'w') as data_file:
     data_file.write(json.dumps(NLBVP_library))import pickle
with open('rainy_benard_NLBVP_alpha{:}_beta{:}_gamma{:}_nz{:}.pickle'.format(α,β,γ,nz), 'w') as data_file:
    pickle.dump(NLBVP_library, data_file, protocol=pickle.HIGHEST_PROTOCOL)import hdf5
for tau_i, sol_set in NLBVP_library.items():
    for k_i, sol in sol_set.items():
        # loop over dictionary items, doing different things for fields and ints
    # use dedalus file handlers
for tau_i, sol_set in NLBVP_library.items():
    for k_i, sol in sol_set.items():
        # loop over dictionary items, saving out b and q
    
# In[15]:


# pick a solution for plotting
NLBVP_sol = NLBVP_library[taus[-1]][ks[-1]]


# In[29]:


value = rh.evaluate()
value.change_scales(1)
mask = (value['g'] >= 1-0.01)
ax = plot_solution(NLBVP_sol, title='NLBVP solution', mask=mask, linestyle='solid')
mask = (value['g'] < 1-0.01)
plot_solution(NLBVP_sol, title='NLBVP solution', mask=mask, linestyle='dashed', ax=ax)
print('zc = {:.3g}'.format(NLBVP_sol['zc']))
print('zc = {:.3g}'.format(find_zc(NLBVP_sol)))


# In[17]:


dz = lambda A: de.Differentiate(A, coords['z'])
dbdz = dz(NLBVP_sol['b']).evaluate()
dqdz = dz(NLBVP_sol['q']).evaluate()
dbdz.change_scales(1)
dqdz.change_scales(1)
NLBVP_sol['rh'].change_scales(1)
fig, ax = plt.subplots(ncols=2)
ax[0].plot(dbdz['g'][0,0,:], z[0,0,:])
ax[0].plot(γ*dqdz['g'][0,0,:], z[0,0,:])
ax[0].plot(NLBVP_sol['rh']['g'][0,0,:], z[0,0,:])
ax[0].axvline(x=1, color='xkcd:grey', linestyle='dashed', alpha=0.5)
ax[0].set_xlabel('$\partial b,~\gamma\partial q,~r_h$')
ax[0].set_ylabel('$z$')
ax[1].plot(dbdz['g'][0,0,:], z[0,0,:], marker='*')
ax[1].plot(γ*dqdz['g'][0,0,:], z[0,0,:], marker='*')
#ax[1].plot(NLBVP_sol['rh']['g'][0,0,:], z[0,0,:])
zc = NLBVP_sol['zc']
#ax[1].axvline(x=1, color='xkcd:grey', linestyle='dashed', alpha=0.5)
ax[1].set_ylim(zc-0.025, zc+0.05)
ax[1].set_xlabel('$\partial b,~\gamma\partial q$')
ax[1].set_title(r'$\gamma$ = {:}, $\beta$ = {:}, $\tau$ = {:.1g}'.format(γ,β, tau['g'][0,0,0]))
ax[1].axhline(y=zc, linestyle='dashed', color='xkcd:dark grey', zorder=0)
ax[0].axhline(y=zc, linestyle='dashed', color='xkcd:dark grey', zorder=0)


# In[20]:


dz = lambda A: de.Differentiate(A, coords['z'])
taus = [k for k in NLBVP_library]
print(taus)
tau_i = taus[-1]
fig, ax = plt.subplots(ncols=2)
ks = np.logspace(2, 3.5, num=8)
linestyles = ['solid']
if len(ks) > 1:
    linestyles.insert(0,'dashed')
if len(ks) > 2:
    linestyles.insert(0, 'dotted')
if len(ks) > 3:
    for i in range(len(ks)-3):
        linestyles.insert(1, 'dashdot')
print(len(ks), len(linestyles))    
print(ks)
for i, k_i in enumerate(NLBVP_library[tau_i]):
    print(k_i)
    sol = NLBVP_library[tau_i][k_i]
    dbdz = dz(sol['b']).evaluate()
    dqdz = dz(sol['q']).evaluate()
    dbdz.change_scales(1)
    dqdz.change_scales(1)
    ax[0].plot(dbdz['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i])
    ax[0].plot(γ*dqdz['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i])
    ax[0].set_xlabel('$\partial b,~\gamma\partial q$')
    ax[0].set_ylabel('$z$')
    ax[1].plot(dbdz['g'][0,0,:], z[0,0,:], marker='*', alpha=0.5, linestyle=linestyles[i])
    ax[1].plot(γ*dqdz['g'][0,0,:], z[0,0,:], marker='*', alpha=0.5, linestyle=linestyles[i], label='q(k={:.1g})'.format(k_i))
    ax[1].set_ylim(0.22, 0.30)
    ax[1].set_xlabel('$\partial b,~\gamma\partial q$')
    ax[0].set_prop_cycle(None)
    ax[1].set_prop_cycle(None)
ax[1].legend()
ax[1].set_title(r'$\gamma$ = {:}, $\beta$ = {:}, $\tau$ = {:.1g}'.format(γ,β, tau_i))


# In[21]:


dz = lambda A: de.Differentiate(A, coords['z'])
taus = [k for k in NLBVP_library]
ks = [k for k in NLBVP_library[taus[-1]]]
print(taus)
fig, ax = plt.subplots(ncols=2, figsize=[10, 4])
for tau_i in taus:
    for i, k_i in enumerate(NLBVP_library[tau_i]):
        sol = NLBVP_library[tau_i][k_i]
        b.change_scales(1)
        q.change_scales(1)
        sol['b'].change_scales(1)
        sol['q'].change_scales(1)
        b['g'] = sol['b']['g']
        q['g'] = sol['q']['g']
        dbdz = dz(b).evaluate()
        dqdz = dz(q).evaluate()
        dbdz.change_scales(1)
        dqdz.change_scales(1)
        b.change_scales(1)
        q.change_scales(1)
        if i == 0:
            p1 = ax[0].plot(b['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i])
            p2 = ax[0].plot(γ*q['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i])
            p3 = ax[0].plot(b['g'][0,0,:]+γ*q['g'][0,0,:], z[0,0,:])
            p1 = ax[1].plot(dbdz['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i])
            p2 = ax[1].plot(γ*dqdz['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i])
        else:
            ax[0].plot(b['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i], color=p1[0].get_color())
            ax[0].plot(γ*q['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i], color=p2[0].get_color())
            ax[0].plot(b['g'][0,0,:]+γ*q['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i], color=p3[0].get_color())
            ax[1].plot(dbdz['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i], color=p1[0].get_color())
            ax[1].plot(γ*dqdz['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i], color=p2[0].get_color())

ax[0].set_xlabel('$b,~\gamma q,~m$')

ax[1].set_xlabel('$\partial b,~\gamma\partial q$')
ax[0].set_ylabel('$z$')
ax[0].set_title(r'$\gamma$ = {:}, $\beta$ = {:}, $\alpha$ = {:}'.format(γ,β,α))
ax[1].set_title(r'$\tau$ = {:.1g}--{:.1g} and $k$ = {:.1g}--{:.1g}'.format(taus[0], taus[-1], ks[0], ks[-1]))


# In[27]:


fig, ax = plt.subplots(ncols=2)
for tau_i in taus:
    zcs=[]
    ks=[]
    zc = None
    linestyles=['dotted', 'dashdot','dashdot','dashed', 'solid']
    for i, k_i in enumerate(NLBVP_library[tau_i]):
        sol = NLBVP_library[tau_i][k_i]
        rh = sol['rh']
        rh.change_scales(1)
        if i == 0:
            p1 = ax[0].plot(rh['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i])
        else:
            ax[0].plot(rh['g'][0,0,:], z[0,0,:], alpha=0.5, linestyle=linestyles[i], color=p1[0].get_color())
        zcs.append(sol['zc'])
        ks.append(k_i)
    ax[1].scatter(ks, zcs)
ax[0].set_xlabel('$r_h$')
ax[0].set_ylabel('z')
ax[1].set_xlabel('k')
ax[1].set_ylabel('$z_c$')
ax[1].set_xscale('log')
ax[1].axhline(y=0.45, color='xkcd:grey', linestyle='dashed', zorder=0)
ax[0].axhline(y=0.45, color='xkcd:grey', linestyle='dashed', zorder=0)
#ax[1].set_yscale('log')
fig.tight_layout()


# In[ ]:


print('diffusion timescale tau')
print('Vallis: {:.2g}'.format(tau_Vallis))
print('us:     {:.2g}'.format(tau['g'][0,0,0]*P))
print('buoyancy timescale tau')
print('Vallis: {:.2g}'.format(tau_Vallis/P))
print('us:     {:.2g}'.format(tau['g'][0,0,0]))


# Hysteresis testing
# ==============
# 
# Let's see if hysteresis is a major problem, by walking down in tau along the high-k space, and then walking a solution back up in tau, starting each time from our libary of solutions.
# 
# Test 1: walk down in tau

# In[ ]:


taus = [k for k in NLBVP_library]
start_tau = taus[0]
stop_tau  = taus[-1]
start_k = 1e5

start_sol = NLBVP_library[start_tau][start_k]
ax = plot_solution(start_sol, title=r'hysteresis test 1: walk $\tau$ = {:.2g}--{:.2g}'.format(start_tau, stop_tau), linestyle='dashed')

b.change_scales(1)
q.change_scales(1)
start_sol['b'].change_scales(1)
start_sol['q'].change_scales(1)
b['g'] = start_sol['b']['g']
q['g'] = start_sol['q']['g']
temp = b - β*z_grid
temp.name = 'T'

qs = np.exp(α*temp)
rh = q*np.exp(-α*temp)

for system in ['subsystems']:
     logging.getLogger(system).setLevel(logging.WARNING)

# Relax on tau
for tau_i in np.logspace(np.log10(start_tau), np.log10(stop_tau), num=4):
    tau['g'] = tau_i
    k = start_k
    solver = problem.build_solver()
    pert_norm = np.inf
    while pert_norm > tol:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info("tau = {:.1g}, k = {:.0g}, L2 err = {:.1g}".format(tau['g'][0,0,0], k, pert_norm))
    sol = {'b':b.copy(), 'q':q.copy(), 'm':(b+γ*q).evaluate().copy(), 'T':temp.evaluate().copy(), 'rh':rh.evaluate().copy()}
    zc = find_zc(sol)
    logger.info('tau = {:.1g}, k = {:.0g}, zc = {:.2g}'.format(tau['g'][0,0,0], k, zc))
NLBVP_sol = {'b':b.copy(), 'q':q.copy(), 'm':(b+γ*q).evaluate().copy(), 'T':temp.evaluate().copy(), 'rh':rh.evaluate().copy()}
plot_solution(NLBVP_sol, linestyle='solid', ax=ax)


# In[ ]:


ax = plot_solution(NLBVP_sol, title=r'hysteresis test 1: walk $\tau$ = {:.2g}--{:.2g}'.format(start_tau, stop_tau), linestyle='solid')
plot_solution(start_sol, linestyle='dashed', ax=ax)

plot_solution(NLBVP_sol, title='hysteresis test 1: final state', linestyle='solid')
plot_solution(start_sol, title='hysteresis test 1: start state', linestyle='dashed')


# Test 2: walk up in tau

# In[ ]:


taus = [k for k in NLBVP_library]
start_tau = taus[-1]
stop_tau  = taus[0]
start_k = 1e5
start_sol = NLBVP_library[start_tau][start_k]
ax = plot_solution(NLBVP_sol, title=r'hysteresis test 2: walk $\tau$ = {:.2g}--{:.2g}'.format(start_tau, stop_tau), linestyle='dashed')

b.change_scales(1)
q.change_scales(1)
start_sol['b'].change_scales(1)
start_sol['q'].change_scales(1)
b['g'] = start_sol['b']['g']
q['g'] = start_sol['q']['g']
temp = b - β*z_grid
temp.name = 'T'

qs = np.exp(α*temp)
rh = q*np.exp(-α*temp)

for system in ['subsystems']:
     logging.getLogger(system).setLevel(logging.WARNING)

# Relax on tau
for tau_i in np.logspace(np.log10(start_tau), np.log10(stop_tau), num=4):
    tau['g'] = tau_i
    k = start_k
    solver = problem.build_solver()
    pert_norm = np.inf
    while pert_norm > tol:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info("tau = {:.1g}, k = {:.0g}, L2 err = {:.1g}".format(tau['g'][0,0,0], k, pert_norm))
    sol = {'b':b.copy(), 'q':q.copy(), 'm':(b+γ*q).evaluate().copy(), 'T':temp.evaluate().copy(), 'rh':rh.evaluate().copy()}
    zc = find_zc(sol)
    logger.info('tau = {:.1g}, k = {:.0g}, zc = {:.2g}'.format(tau['g'][0,0,0], k, zc))
NLBVP_sol = {'b':b.copy(), 'q':q.copy(), 'm':(b+γ*q).evaluate().copy(), 'T':temp.evaluate().copy(), 'rh':rh.evaluate().copy()}
plot_solution(NLBVP_sol, linestyle='solid', ax=ax)


# In[ ]:


ax = plot_solution(NLBVP_sol, title=r'hysteresis test 2: walk $\tau$ = {:.2g}--{:.2g}'.format(start_tau, stop_tau), linestyle='solid')

plot_solution(start_sol, linestyle='dashed', ax=ax)

plot_solution(NLBVP_sol, title='hysteresis test 2: final state', linestyle='solid')
plot_solution(start_sol, title='hysteresis test 2: start state', linestyle='dashed')


# Summary on hysterisis
# ==================
# At loose tolerances (`tol=1e-2`), we saw evidence of hysteretic behaviour.  Walking down in tau at fixed k (`k=1e5`) lead to different solutions than walking up in tau.  At loose tolerances, `zc(tau, k)` shows structure in both tau and k.
# 
# If the tolerances are sufficiently tight (`tol=1e-3`), all evidence of this hysteretic behaviour goes away.  Additionally, `zc(tau, k) = zc(tau)`, e.g. there's only structure in the tau dimension.
# 
# At `tol=1e-3`, the solution time to build the initial library is about 10 minutes on a Macbook Pro M1, which is about double the time at looser tolerances, but very reasonable.

# WIP
# ======
# Everything below here works for saturated atmospheres, but we need to figure out how to do the unsaturated lower atmosphere and matching $z_c$ conditions.
# 
# 
# Analytic solutions
# ---------------------
# For saturated atmospheres, we can construct an analytic function using Lambert W functions and following the discussion in section 5.1.
# 
# Things are a bit more complicated in a partially unsaturated atmosphere.  In that atmosphere, we need to construct a linear $q$ and $b$ profile until we hit $z_c$ where $q(z_c) = q_s(z_c)$, and then we proceed with the Lambert W solution.
# 
# Assume we start at $q(z=0) = q_0$ and $b(z=0)=b_0$, and that both $q$ and $b$ have linear profiles:
# \begin{align}
#     q(z) & = q_0 + Q z \\
#     b(z) & = b_0 + B z \\
#     T(z) & = b(z) - \beta z = b_0 + (B-\beta) z
# \end{align}
# They note that at $z=z_c$, $q$, $b$, $\partial_z q$ and $\partial_z b$ are all continuous, and this can be used (numerically) to solve for $z_c$.  Okay.
# 
# Meanwhile,
# \begin{align}
# q_s(z) = \exp{(\alpha T)} = \exp{(\alpha (b_0 + (B-\beta) z)}
# \end{align}
# and
# \begin{align}
#     q(z_c) &= q_s(z_c) \\
#     q_0 + Q z_c &= \exp{(\alpha (b_0 + (B-\beta) z_c)}
# \end{align}
# We could rootfind on this for z_c if we knew $Q$ and $B$.  Hmm...
# 
# 
# At $z_c$ we know:
# 
# from the bottom:
# \begin{align}
# b(z_c-) &= b_0 + B z_c \\
# \partial b(z_c-) &= B
# \end{align}
# (whoa, B!=0, so that's inconsistent with 5.14, which said that $\partial b(z_c) = 0$)
# 
# oh, okay, (5.14) actually says:
# \begin{align}
# \partial q(z_c+) - \partial q(z_c-) = 0
# \end{align}
# ok!
# 
# And $T = b - \beta z$ (eq 4.9) so $b = T + \beta z$, and the upper solution is:
# \begin{align}
# T(z_c+) & = P + (Q-\beta) z_c+ - \frac{W(\alpha \gamma \exp(\alpha(P+(Q-\beta) z_c+)))}{\alpha}
# \end{align}
# or
# \begin{align}
# b(z_c+) & = P + Q z_c+ - \frac{W(\alpha \gamma \exp(\alpha(P+(Q-\beta) z_c+)))}{\alpha} \\
# \partial b(z_c+) & = Q - \partial \frac{W(\alpha \gamma \exp(\alpha(P+(Q-\beta) z_c+)))}{\alpha}
# \end{align}
# 
# Now, where are things complicated.

# In[ ]:


from scipy.special import lambertw as W
def compute_analytic(z_in):
    z = dist.Field(bases=zb)
    z['g'] = z_in

    b1 = 0
    b2 = β + ΔT
    q1 = q_surface
    q2 = np.exp(α*ΔT)

    P = b1 + γ*q1
    Q = ((b2-b1) + γ*(q2-q1))
    
    C = P + (Q-β)*z['g']
    
    m = (P+Q*z).evaluate()
    T = dist.Field(bases=zb)
    T['g'] = C - W(α*γ*np.exp(α*C)).real/α
    b = (T + β*z).evaluate()
    q = ((m-b)/γ).evaluate()
    rh = (q*np.exp(-α*T)).evaluate()
    return {'b':b, 'q':q, 'm':m, 'T':T, 'rh':rh}

analytic_sol = compute_analytic(z)
analytic_sol['rh'].change_scales(1)
mask = (analytic_sol['rh']['g'] >= 1)
ax = plot_solution(analytic_sol, title='Lambert W solution', mask=mask, linestyle='solid')
#mask = (analytic_solution['rh']['g'] < 1)
#plot_solution(solution=analytic_solution, mask=mask, linestyle='dashed', ax=ax)ax = plot_solution(NLBVP_sol, title='compare NLBVP (solid) vs W (dashed) solution', mask=mask, linestyle='solid')
plot_solution(analytic_sol, linestyle='dashed', ax=ax)
