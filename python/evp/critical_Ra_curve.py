"""
Dedalus script for determining instability of static drizzle solutions to the Rainy-Benard system of equations.  This script computes curves of growth at discrete kx, scanning a variety of Rayleigh numbers.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves EVPs for an existing atmospheres, solved for by scripts in the nlbvp section.

Roberts, G.O., 1972,
``Dynamo action of fluid motions with two-dimensional periodicity''

Usage:
    convective_onset.py [options]

Options:
    --nondim=<n>      Non-Nondimensionalization [default: buoyancy]

    --alpha=<alpha>      Alpha parameter [default: 3]
    --gamma_min=<gmn>    Gamma parameter [default: 0.15]
    --gamma_max=<gmx>    Gamma parameter [default: 0.4]
    --gamma_n=<gn>       Number of gamma points [default: 10]
    --beta=<beta>        Beta parameter  [default: 1.2]

    --tau=<tau>          Tau parameter [default: 1e-3]
    --k=<k>              Tanh width of phase change [default: 1e3]

    --min_Ra=<minR>   Minimum Rayleigh number to sample [default: 1e3]
    --max_Ra=<maxR>   Maximum Rayleigh number to sample [default: 1e5]
    --num_Ra=<nRa>    How many Rayleigh numbers to sample [default: 21]

    --num_k=<nk>      How many kxs to sample [default: 11]

    --top-stress-free     Stress-free upper boundary
    --stress-free         Stress-free both boundaries

    --nz=<nz>         Number of coeffs to use in eigenvalue search [default: 64]
    --target=<targ>   Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>     Target number of eigenvalues to search for [default: 20]

    --dense           Solve densely for all eigenvalues (slow)

    --verbose         Show plots on screen
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)

import numpy as np
import dedalus.public as de
import h5py

from docopt import docopt
args = docopt(__doc__)
dtype = np.complex128

N_evals = int(float(args['--eigs']))
target = float(args['--target'])

tau_in = float(args['--tau'])
k = float(args['--k'])

q_surface = 1
nz = int(args['--nz'])

α = float(args['--alpha'])
β = float(args['--beta'])
γmn = float(args['--gamma_min'])
γmx = float(args['--gamma_max'])
γn  = int(float(args['--gamma_n']))
γs = np.linspace(γmn, γmx, num=γn, dtype=dtype)

logger.info('α={:}, β={:}, γ={:}, tau={:}, k={:}'.format(α,β,γs,tau_in, k))

nz = int(float(args['--nz']))

ΔT = -1

from scipy.special import lambertw as W
def compute_analytic(z_in, γ):
    z = dist.Field(bases=zb)
    z['g'] = z_in

    b1 = 0
    b2 = β + ΔT
    q1 = q_surface
    q2 = np.exp(α*ΔT)

    P = b1 + γ*q1
    Q = ((b2-b1) + γ*(q2-q1))

    C = P + (Q-β)*z['g']
    m = (Q*z).evaluate()
    m = (P+Q*z).evaluate()
    T = dist.Field(bases=zb)
    T['g'] = C - W(α*γ*np.exp(α*C)).real/α
    b = (T + β*z).evaluate()
    q = ((m-b)/γ).evaluate()
    rh = (q*np.exp(-α*T)).evaluate()
    return {'b':b, 'q':q, 'm':m, 'T':T, 'rh':rh}

Prandtlm = 1
Prandtl = 1

Lz = 1
coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype)
dealias = 1
zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
z = zb.local_grid(1)
zd = zb.local_grid(dealias)

b0 = dist.Field(name='b0', bases=zb)
q0 = dist.Field(name='q0', bases=zb)
γ = dist.Field(name='gamma')

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

lift = lambda A, n: de.Lift(A, zb, n)

ex, ey, ez = coords.unit_vector_fields(dist)

H = lambda A: 0.5*(1+np.tanh(k*A))

z_grid = dist.Field(name='z_grid', bases=zb)
z_grid['g'] = z

T0 = dist.Field(name='T', bases=zb)
qs0 = dist.Field(name='T', bases=zb)

tau = dist.Field(name='tau')
kx = dist.Field(name='kx')
Rayleigh = dist.Field(name='Ra_c')

# follows Roberts 1972 convention, eq 1.1, 2.8
dx = lambda A: 1j*kx*A # 1-d mode onset
dy = lambda A: 0*A # flexibility to add 2-d mode if desired

grad = lambda A: de.Gradient(A, coords) + dx(A)*ex + dy(A)*ey
div = lambda A:  de.div(A) + dx(A@ex) + dy(A@ey)
lap = lambda A: de.lap(A) + dx(dx(A)) + dy(dy(A))
trans = lambda A: de.TransposeComponents(A)

e = grad(u) + trans(grad(u))
vars = [p, u, b, q, τp, τu1, τu2, τb1, τb2, τq1, τq2]
# fix Ra, find omega
dt = lambda A: ω*A
ω = dist.Field(name='ω')
problem = de.EVP(vars, eigenvalue=ω, namespace=locals())

nondim = args['--nondim']
if nondim == 'diffusion':
    P = 1                      #  diffusion on buoyancy. Always = 1 in this scaling.
    S = Prandtlm               #  diffusion on moisture  k_q / k_b
    PdR = Prandtl              #  diffusion on momentum
    PtR = Prandtl*Rayleigh     #  Prandtl times Rayleigh = buoyancy force
elif nondim == 'buoyancy':
    P = (Rayleigh * Prandtl)**(-1/2)         #  diffusion on buoyancy
    S = (Rayleigh * Prandtlm)**(-1/2)        #  diffusion on moisture
    PdR = (Rayleigh/Prandtl)**(-1/2)         #  diffusion on momentum
    PtR = 1
    #tau_in /=                     # think through what this should be
else:
    raise ValueError('nondim {:} not in valid set [diffusion, buoyancy]'.format(nondim))

tau['g'] = tau_in

# sech = lambda A: 1/np.cosh(A)
# scrN = (H(q0 - qs0) + 1/2*(q0 - qs0)*k**2*sech(k*(q0 - qs0))**2).evaluate()
# scrN = (H(q0 - qs0) + 1/2*(q0 - qs0)*k*(1-(np.tanh(k*(q0 - qs0)))**2)).evaluate()
scrN = dist.Field(bases=zb)
scrN['g'] = 1 #0.5
scrN.name='scrN'
grad_b0 = dist.VectorField(coords, name='grad_b0', bases=zb)
grad_q0 = dist.VectorField(coords, name='grad_q0', bases=zb)
#
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
solver = problem.build_solver()

dlog = logging.getLogger('subsystems')
dlog.setLevel(logging.WARNING)

# fix Ra, find omega
def compute_growth_rate(kx_i, Ra_i):
    kx['g'] = kx_i
    Rayleigh['g'] = Ra_i
    if args['--dense']:
        solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
        solver.eigenvalues = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
    else:
        solver.solve_sparse(solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
    i_evals = np.argsort(solver.eigenvalues.real)
    evals = solver.eigenvalues[i_evals]
    peak_eval = evals[-1]
    # choose convention: return the positive complex mode of the pair
    if peak_eval.imag < 0:
        peak_eval = np.conj(peak_eval)
    return peak_eval

def peak_growth_rate(*args):
    rate = compute_growth_rate(*args)
    # flip sign so minimize finds maximum
    return -1*rate.real

import scipy.optimize as sciop

growth_rates = {}
Ras = np.geomspace(float(args['--min_Ra']),float(args['--max_Ra']),num=int(float(args['--num_Ra'])))
kxs = np.logspace(0, 1, num=int(float(args['--num_k'])))
print(Ras)

crit_Ras = []
crit_ks = []
crit_σ_Rs = []
crit_σ_Is = []

for γ_i in γs:
    analytic_sol = compute_analytic(z, γ_i)

    logger.info('setting background from analytic solution, γ={:}'.format(γ_i))
    b0.change_scales(dealias)
    q0.change_scales(dealias)
    z_grid.change_scales(dealias)
    T0.change_scales(dealias)
    qs0.change_scales(dealias)
    grad_b0.change_scales(dealias)
    grad_q0.change_scales(dealias)

    b0['g'] = analytic_sol['b']['g']
    q0['g'] = analytic_sol['q']['g']
    γ['g'] = γ_i
    T0['g'] = b0['g'] - β*z_grid['g']
    qs0['g'] = np.exp(α*T0['g'])
    grad_b0['g'] = grad(b0).evaluate()['g']
    grad_q0['g'] = grad(q0).evaluate()['g']

    for Ra_i in Ras:
        σ = []
        for kx_i in kxs:
            σ_i = compute_growth_rate(kx_i, Ra_i)
            σ.append(σ_i)
            logger.info('Ra = {:.2g}, kx = {:.2g}, σ = {:.2g}'.format(Ra_i, kx_i, σ_i))
        growth_rates[Ra_i] = np.array(σ)


    bounds = sciop.Bounds(lb=1, ub=10)

    peaks = {'σ':[], 'k':[], 'Ra':[]}
    for Ra in growth_rates:
        σ = growth_rates[Ra]
        peak_i = np.argmax(σ)
        kx_i = kxs[peak_i]
        result = sciop.minimize(peak_growth_rate, kx_i, args=(Ra), bounds=bounds, method='Nelder-Mead', tol=1e-5)
        # obtain full complex rate
        σ = compute_growth_rate(result.x[0], Ra)
        logger.info('peak search: start at Ra = {:.4g}, kx = {:.4g}, found σ_max = {:.2g},{:.2g}i, kx = {:.4g}'.format(Ra, kx_i, σ.real, σ.imag, result.x[0]))
        peaks['σ'].append(σ)
        peaks['k'].append(result.x[0])
        peaks['Ra'].append(Ra)

    peaks['σ'] = np.array(peaks['σ'])
    peaks['k'] = np.array(peaks['k'])
    peaks['Ra'] = np.array(peaks['Ra'])

    from scipy.interpolate import interp1d
    f_σR_i = interp1d(peaks['σ'].real, peaks['k']) #inverse
    f_σR = interp1d(peaks['k'], peaks['σ'].real)
    f_σI_i = interp1d(peaks['σ'].imag, peaks['k']) #inverse
    f_σI = interp1d(peaks['k'], peaks['σ'].imag)

    # to find critical Ra
    f_σR_Ra_i = interp1d(peaks['σ'].real, peaks['Ra'])
    f_σR_Ra = interp1d(peaks['Ra'], peaks['σ'].real)
    f_σI_Ra = interp1d(peaks['Ra'], peaks['σ'].imag)
    f_k_Ra = interp1d(peaks['Ra'], peaks['k'])

    peak_ks = np.geomspace(np.min(peaks['k']), np.max(peaks['k']))

    crit_Ra = f_σR_Ra_i(0)
    crit_k = f_k_Ra(crit_Ra)
    crit_σ_R = f_σR_Ra(crit_Ra)
    crit_σ_I = f_σI_Ra(crit_Ra)
    logger.info('Critical point, based on interpolation:')
    logger.info('Ra = {:}, k = {:}'.format(crit_Ra, crit_k))
    logger.info('σ = {:}, {:}i'.format(crit_σ_R, crit_σ_I))
    crit_Ras.append(crit_Ra)
    crit_ks.append(crit_k)
    crit_σ_Rs.append(crit_σ_R)
    crit_σ_Is.append(crit_σ_I)

data = h5py.File('{:s}'.format('critical_Ra_alpha{:}_beta{:}.h5'.format(α,β)), 'a')
data['γ'] = γs
data['Ra_c'] = np.array(crit_Ras)
data['k_c'] = np.array(crit_ks)
data['σ_R'] = np.array(crit_σ_Rs)
data['σ_I'] = np.array(crit_σ_Is)
data.close()
