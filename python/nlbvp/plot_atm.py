"""
Script for plotting atmosphere properties for unsaturated/saturated atmospheres.

This script plots NLBVP solutions for unsaturated atmospheres, which corresponds to Vallis et al 2019, Figure 2.

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

Usage:
    plot_atm.py <cases>... [options]

Options:
    <cases>         Case (or cases) to plot results from

    --epsilon=<e>   Epsilon to control search for zc [default: 1e-3]
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
    ax[1].axvline(x=1, linestyle='dotted')
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

def find_zc(sol, ε=1e-3, root_finding = 'inverse'):
    rh = sol['rh']
    z = sol['z']
    nz = z.shape[0]
    if root_finding == 'inverse':
        # invert the relationship and use interpolation to find where r_h = 1-ε (approach from below)
        f_i = interp1d(rh, z, bounds_error=False, fill_value=np.NaN) #inverse
        zc = f_i(1-ε)
    elif root_finding == 'discrete':
        # crude initial emperical zc; look for where rh-1 ~ 0, in lower half of domain.
        zc = z[np.argmin(np.abs(rh[0:int(nz/2)]-1))]
#    if zc is None:
#        zc = 0.2
#    zc = newton(f, 0.2)
    return zc

if __name__=="__main__":
    from docopt import docopt
    args = docopt(__doc__)

    import h5py
    import dedalus.public as de
    logger = logging.getLogger(__name__)

    ε = float(args['--epsilon'])

    for case in args['<cases>']:
        f = h5py.File(case+'/drizzle_sol/drizzle_sol_s1.h5', 'r')
        sol = {}
        for task in f['tasks']:
            sol[task] = f['tasks'][task][0,0,0][:]
        sol['z'] = f['tasks']['b'].dims[3][0][:]

        mask = (sol['rh'] >= 1-ε)
        fig, ax = plot_solution(sol, mask=mask, linestyle='solid')
        mask = (sol['rh'] < 1-ε)
        plot_solution(sol, mask=mask, linestyle='dashed', ax=ax)
        fig.tight_layout()
        fig.savefig(case+'/atm.png', dpi=300)
        fig.savefig(case+'/atm.pdf')
        zc = find_zc(sol, ε=ε)
        logger.info('tau = {:.1g}, k = {:.0g}, zc = {:.2g}'.format(sol['tau'][0], sol['k'][0], zc))
        plt.close(fig)
