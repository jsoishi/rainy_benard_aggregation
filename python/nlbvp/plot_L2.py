"""
Script for plotting atmosphere properties for unsaturated/saturated atmospheres.

This script plots NLBVP solutions for unsaturated atmospheres, which corresponds to Vallis et al 2019, Figure 2.

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

Usage:
    plot_atm.py <cases>... [options]

Options:
    <cases>         Case (or cases) to plot results from

    --gamma=<g>     γ value [default: 0.19]
    --beta=<b>      β value [default: 1.1]
    --alpha=<a>     α value [default: 3]

    --Legendre
    --nz=<nz>       Z resolution [default: 384]
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np

from scipy.special import lambertw as W
def unsaturated(dist, zb, β, γ, zc, Tc,
                     dealias=1, q0=0.6, α=3):

    z = dist.Field(bases=zb)
    z['g'] = zb.local_grid(1)
    z.change_scales(dealias)

    q = dist.Field(bases=zb)
    T = dist.Field(bases=zb)
    q.change_scales(dealias)
    T.change_scales(dealias)

    b1 = 0
    b2 = β + ΔT
    q1 = q0
    q2 = np.exp(α*ΔT)

    bc = Tc + β*zc
    qc = np.exp(α*Tc)

    P = bc + γ*qc
    Q = ((b2-bc) + γ*(q2-qc))

    C = P + Q*(z['g']-zc)/(1-zc) - β*z['g']

    mask = (z['g']>=zc)
    T['g'][mask] = C[mask] - W(α*γ*np.exp(α*C[mask])).real/α
    T['g'][~mask] = Tc*z['g'][~mask]/zc

    b = (T + β*z).evaluate()
    q['g'][mask] = np.exp(α*T['g'][mask])
    q['g'][~mask] = q1 + (qc-q1)*z['g'][~mask]/zc
    m = (b + γ*q).evaluate()
    rh = (q*np.exp(-α*T)).evaluate()
    return {'b':b, 'q':q, 'm':m, 'T':T, 'rh':rh, 'z':z, 'γ':γ}

if __name__=="__main__":
    from docopt import docopt
    args = docopt(__doc__)

    import pandas as pd
    import h5py
    import dedalus.public as de
    logger = logging.getLogger(__name__)

    min_tau = np.inf
    max_tau = 0

    γ = float(args['--gamma'])
    β = float(args['--beta'])
    α = float(args['--alpha'])
    ΔT = -1

    nz = int(float(args['--nz']))
    Lz = 1
    dealias=1 #3/2
    dtype = np.float64
    coords = de.CartesianCoordinates('z')
    dist = de.Distributor(coords, dtype=dtype)
    if args['--Legendre']:
        zb = de.Legendre(coords.coords[-1], size=nz, bounds=(0, Lz), dealias=dealias)
    else:
        zb = de.ChebyshevT(coords.coords[-1], size=nz, bounds=(0, Lz), dealias=dealias)
    if γ == 0.3:
        zc_analytic = 0.4832893544084419
        Tc_analytic = -0.4588071140209613
    elif γ == 0.19:
        zc_analytic = 0.4751621541611023
        Tc_analytic = -0.4588071140209616
    else:
        raise ValueError("γ = {:} not yet supported".format(γ))

    zc = zc_analytic
    Tc = Tc_analytic
    analytic = unsaturated(dist, zb, β, γ, zc, Tc, α=α, dealias=dealias)

    integ = lambda A: de.Integrate(A, 'z')
    def compute_L2_err(sol, analytic):
        return (integ(np.abs(sol-analytic))/integ(analytic)).evaluate()['g'][0]

    q = dist.Field(bases=zb)
    b = dist.Field(bases=zb)
    rh = dist.Field(bases=zb)

    data = {} #{'tau':[],'k':[],'zc':[]}
    for case in args['<cases>']:
        try:
            f = h5py.File(case+'/drizzle_sol/drizzle_sol_s1.h5', 'r')
            sol = {}
            for task in f['tasks']:
                sol[task] = f['tasks'][task][0,0,0][:]
            sol['z'] = f['tasks']['b'].dims[3][0][:]

            tau = sol['tau'][0]
            k = sol['k'][0]
            γ = sol['γ'][0]
            q['g'] = sol['q']
            b['g'] = sol['b']
            rh['g'] = sol['rh']

            L2_q = compute_L2_err(q, analytic['q'])
            L2_b = compute_L2_err(b, analytic['b'])
            L2_rh = compute_L2_err(rh, analytic['rh'])
            L2_results = "L2:  q = {:.2g}, b={:.2g}, rh={:.2g}".format(L2_q, L2_b, L2_rh)

            if tau in data:
                data[tau]['k'].append(k)
                data[tau]['L2_q'].append(L2_q)
                data[tau]['L2_b'].append(L2_b)
                data[tau]['L2_rh'].append(L2_rh)
            else:
                data[tau] = {'k':[k], 'L2_q':[L2_q], 'L2_b':[L2_b], 'L2_rh':[L2_rh]}
                min_tau = min(tau, min_tau)
                max_tau = max(tau, max_tau)
            logger.debug('{:}:'.format(case))
            logger.info('tau = {:.1g}, k = {:.0g}, {:s}'.format(tau, k,  L2_results))
        except:
            logger.warning('error in case {:}'.format(case))
    for tau in data:
        data[tau]['L2_q'] = np.array(data[tau]['L2_q'])
        data[tau]['L2_b'] = np.array(data[tau]['L2_b'])
        data[tau]['L2_rh'] = np.array(data[tau]['L2_rh'])

    import matplotlib.colors as colors
    norm = colors.LogNorm(vmin=min_tau, vmax=max_tau)

    for quant in ['q', 'b', 'rh']:
        fig, ax = plt.subplots(figsize=[6,6/1.6])
        for tau in data:
            p = ax.scatter(data[tau]['k'], data[tau]['L2_{:s}'.format(quant)], c=tau*np.ones_like(data[tau]['k']), norm=norm)
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$|{:s}_c - {:s}_a|/|{:s}_a|$'.format(quant,quant,quant))
        fig.colorbar(mappable=p, ax=ax, orientation='horizontal', location='top', label=r'$\tau$', norm=norm)
        ax.set_xscale('log')
        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(case+'/../L2_{:s}.png'.format(quant), dpi=300)
