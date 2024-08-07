"""
Script for plotting atmosphere properties for unsaturated/saturated atmospheres.

This script plots NLBVP solutions for unsaturated atmospheres, which corresponds to Vallis et al 2019, Figure 2.

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

Usage:
    plot_atm.py <cases>... [options]

Options:
    <cases>         Case (or cases) to plot results from

    --method=<m>    Search method [default: log_newton]
    --epsilon=<e>   Epsilon to control search for zc [default: 1e-5]

    --show_discrete
"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np

def plot_solution(solution, title=None, mask=None, linestyle=None, ax=None):
    b = solution['b']
    q = solution['q']
    m = solution['m']
    T = solution['T']
    rh = solution['rh']

    z = solution['z']
    γ = solution['γ']

    if mask is None:
        mask = np.ones_like(z, dtype=bool)
    if ax is None:
        fig, ax = plt.subplots(ncols=2, sharey=True)
        markup = True
        return_fig = True
    else:
        for axi in ax:
            axi.set_prop_cycle(None)
        markup = False
        return_fig = False

    ax[0].plot(b[mask],z[mask], label='$b$', linestyle=linestyle)
    ax[0].plot(γ*q[mask],z[mask], label='$\gamma q$', linestyle=linestyle)
    ax[0].plot(m[mask],z[mask], label='$b+\gamma q$', linestyle=linestyle)

    ax[1].plot(T[mask],z[mask], label='$T$', linestyle=linestyle)
    ax[1].plot(q[mask],z[mask], label='$q$', linestyle=linestyle)
    ax[1].plot(rh[mask],z[mask], label='$r_h$', linestyle=linestyle)

    if markup:
        ax[1].legend()
        ax[0].legend()
        ax[0].set_ylabel('z')
        if title:
            ax[0].set_title(title)
    if return_fig:
        return fig, ax
    else:
        return ax

from scipy.optimize import newton
from scipy.interpolate import interp1d
from numpy.polynomial import Chebyshev as P

def find_zc(sol, ε=1e-4, deg=256, root_finding = 'log_newton'):
    rh = sol['rh']
    z = sol['z']
    nz = z.shape[0]
    zc0 = z[np.argmin(np.abs(rh[0:int(nz*3/4)]-(1-ε)))]
    if root_finding == 'inverse':
        # invert the relationship and use interpolation to find where r_h = 1-ε (approach from below)
        #f_i = interp1d(rh, z) #inverse
        #mask = slice(0,int(nz*3/4))
        y = rh
        f_i = P.fit(y,z,deg)
        print(f'deg = {deg:4d} L2 = {np.sum(np.abs(f_i(y)-z)):.3g}' )
        zc = f_i(1-ε)
    elif root_finding == 'discrete':
        # crude initial emperical zc; look for where rh-1 ~ 0, in lower half of domain.
        zc = zc0
    elif root_finding == 'newton':
        f = interp1d(z, rh-(1-ε))
        zc = newton(f, zc0)
    elif root_finding == 'log_newton':
        # find when rh+ε=1, or rh=1-ε
        #f = interp1d(z, np.log(rh+ε), kind='cubic')
        y = np.log(rh+ε)
        f = P.fit(z,y,deg)
        print(f'deg = {deg:4d} L2 = {np.sum(np.abs(f(z)-y)):.3g}' )
        zc = newton(f, zc0)
    else:
        raise ValueError('search method {:} not in [inverse, discrete]'.format(root_finding))
    return zc

if __name__=="__main__":
    from docopt import docopt
    args = docopt(__doc__)

    import pandas as pd
    import h5py
    import dedalus.public as de
    logger = logging.getLogger(__name__)

    ε = float(args['--epsilon'])

    min_tau = np.inf
    max_tau = 0

    data = {} #{'tau':[],'k':[],'zc':[]}
    for case in args['<cases>']:
        try:
            f = h5py.File(case+'/drizzle_sol/drizzle_sol_s1.h5', 'r')
            sol = {}
            for task in f['tasks']:
                sol[task] = f['tasks'][task][0,0,0][:]
            sol['z'] = f['tasks']['b'].dims[3][0][:]
            zc = find_zc(sol, ε=ε, root_finding=args['--method'])
            zc_discrete = find_zc(sol, ε=ε, root_finding='discrete')
            tau = sol['tau'][0]
            k = sol['k'][0]
            γ = sol['γ'][0]
            if tau in data:
                data[tau]['k'].append(k)
                data[tau]['zc'].append(zc)
                data[tau]['zc_discrete'].append(zc_discrete)
            else:
                data[tau] = {'k':[k], 'zc':[zc], 'zc_discrete':[zc_discrete]}
                min_tau = min(tau, min_tau)
                max_tau = max(tau, max_tau)
            logger.debug('{:}:'.format(case))
            logger.info('tau = {:.1g}, k = {:.0g}, zc = {:.2g}, {:.2g}'.format(tau, k, zc, zc_discrete))
        except:
            logger.warning('error in case {:}'.format(case))
    for tau in data:
        data[tau]['zc'] = np.array(data[tau]['zc'])

    from analytic_zc import f_zc
    zc_analytic = f_zc()(γ)
    print(f'analytic zc = {zc_analytic:.4g}')

    import matplotlib.colors as colors
    norm = colors.LogNorm(vmin=min_tau, vmax=max_tau)

    fig, ax = plt.subplots(figsize=[6,6/1.6])
    for tau in data:
        p = ax.scatter(data[tau]['k'], data[tau]['zc'], c=tau*np.ones_like(data[tau]['k']), norm=norm, alpha=0.5)
        if args['--show_discrete']:
            ax.scatter(data[tau]['k'], data[tau]['zc_discrete'], c=tau*np.ones_like(data[tau]['k']), norm=norm, alpha=0.3, marker='s')
    ax.axhline(y=zc_analytic, linestyle='dashed', color='xkcd:dark grey', alpha=0.5)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$z_c$')
    fig.colorbar(mappable=p, ax=ax, orientation='horizontal', location='top', label=r'$\tau$', norm=norm)
    fig.tight_layout()
    ax.set_xscale('log')
    fig.savefig(case+'/../zc.png', dpi=300)

    fig, ax = plt.subplots(figsize=[6,6/1.6])
    for tau in data:
        p = ax.scatter(data[tau]['k'], data[tau]['zc']-zc_analytic, c=tau*np.ones_like(data[tau]['k']), norm=norm)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$z_c - z_{c,a}$')
    fig.colorbar(mappable=p, ax=ax, orientation='horizontal', location='top', label=r'$\tau$', norm=norm)
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(case+'/../delta_zc_log.png', dpi=300)

    fig, ax = plt.subplots(figsize=[6,6/1.6])
    for tau in data:
        p = ax.scatter(data[tau]['k'], np.abs(data[tau]['zc']-zc_analytic)/zc_analytic, c=tau*np.ones_like(data[tau]['k']), norm=norm)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$|z_c - z_{c,a}|/z_{c,a}$')
    fig.colorbar(mappable=p, ax=ax, orientation='horizontal', location='top', label=r'$\tau$', norm=norm)
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(case+'/../L2_zc_log.png', dpi=300)
