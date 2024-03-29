"""
Dedalus script for solving static drizzle solutions to the Rainy-Benard system of equations.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script creates analytic solutions unsaturated atmospheres, which corresponds to Vallis et al 2019, Figure 1 and 2.


Usage:
    analytic_atmosphere.py [options]

Options:
    --alpha=<alpha>      Alpha parameter [default: 3]
    --gamma=<gamma>      Gamma parameter [default: 0.19]
    --beta=<beta>        Beta parameter  [default: 1.15]

    --q0=<q0>            Lower moisture value for unsaturated [default: 0.6]
"""
from scipy.special import lambertw as W
import numpy as np

ΔT = -1

def saturated(dist, zb, β, γ,
              dealias=1, q0=1, α=3):

    z = dist.Field(bases=zb)
    z['g'] = dist.local_grid(zb)

    b1 = 0
    b2 = β + ΔT
    q1 = q0
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
    return {'b':b, 'q':q, 'm':m, 'T':T, 'rh':rh, 'z':z, 'γ':γ}

def saturated_VPT19(dist, zb, β, γ,
              dealias=1, q0=1, α=3,
              K2=4e-10, T0=5.5):

    z = dist.Field(bases=zb)
    z['g'] = dist.local_grid(zb)

    b1 = T0
    b2 = T0 + β + ΔT
    q1 = K2*np.exp(α*T0)*q0
    q2 = K2*np.exp(α*T0)*np.exp(α*ΔT)
    M = γ/3.8e-3

    P = b1 + M*q1
    Q = ((b2-b1) + M*(q2-q1))

    C = P + (Q-β)*z['g']

    m = (P+Q*z).evaluate()

    T = dist.Field(bases=zb)
    T['g'] = C - W(α*M*K2*np.exp(α*C)).real/α
    b = (T + β*z).evaluate()
    q = ((m-b)/M).evaluate()
    rh = (q*np.exp(-α*T)).evaluate()
    return {'b':b, 'q':q, 'm':m, 'T':T, 'rh':rh, 'z':z, 'γ':γ}

def unsaturated(dist, zb, β, γ, zc, Tc,
                dealias=1, q0=0.6, α=3):

    z = dist.Field(bases=zb)
    z['g'] = dist.local_grid(zb)
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

    # if zc > 0:
    #     bc = Tc + β*zc
    #     qc = np.exp(α*Tc)
    # else:
    #     bc = b1
    #     qc = q1

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

def plot_solution(solution, title=None, mask=None, markup=None,
                  linestyle=None, ax=None, **kwargs):
    b = solution['b']['g']
    q = solution['q']['g']
    m = solution['m']['g']
    T = solution['T']['g']
    rh = solution['rh']['g']

    z = solution['z']['g']
    γ = solution['γ']

    if mask is None:
        mask = np.ones_like(z, dtype=bool)
    if ax is None:
        fig, ax = plt.subplots(ncols=2, sharey=True, figsize=[1.8*6,1.8*6/1.6])
        if markup is None:
            markup = True
        return_fig = True
    else:
        for axi in ax:
            axi.set_prop_cycle(None)
        if markup is None:
            markup = False
        return_fig = False

    ax[0].plot(b[mask],z[mask], label='$b$', linestyle=linestyle, color='xkcd:dark red', **kwargs)
    ax[0].plot(γ*q[mask],z[mask], label='$\gamma q$', linestyle=linestyle, color='xkcd:french blue', **kwargs)
    ax[0].plot(m[mask],z[mask], label='$m$', linestyle=linestyle, color='xkcd:viridian', **kwargs)

    ax[1].plot(T[mask],z[mask], label='$T$', linestyle=linestyle, color='xkcd:electric pink', **kwargs)
    ax[1].plot(q[mask],z[mask], label='$q$', linestyle=linestyle, color='xkcd:french blue', **kwargs)
    ax[1].plot(rh[mask],z[mask], label='$r_h$', linestyle=linestyle, color='xkcd:perrywinkle', **kwargs)

    ax[1].axvline(x=1, linestyle='dotted', color='xkcd:dark grey', alpha=0.5)
    if markup:
        ax[0].set_ylabel('z')
        ax[0].legend(loc='lower right')
        ax[1].legend()
        if title:
            ax[0].set_title(title)

    if return_fig:
        return fig, ax
    else:
        return ax

if __name__=="__main__":
    import matplotlib.pyplot as plt
    #plt.style.use("../../prl.mplstyle")
    plt.style.use("prl.mplstyle")
    import dedalus.public as de

    from docopt import docopt
    args = docopt(__doc__)

    α = float(args['--alpha'])
    β = float(args['--beta'])
    γ = float(args['--gamma'])

    q0 = float(args['--q0'])

    dealias = 1
    nz = 128
    dtype = np.float64
    Lz = 1

    coords = de.CartesianCoordinates('x', 'y', 'z')
    dist = de.Distributor(coords, dtype=dtype)
    zb = de.ChebyshevT(coords.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)

    sol = saturated_VPT19(dist, zb, β, γ, α=α, dealias=dealias)
    fig, ax = plot_solution(sol)
    ax[0].set_xlim(5.5,5.8)
    ax[1].set_xlim(4.5,5.5)
    fig.tight_layout()
    fig.savefig(f'analytic_VPT19_alpha{α}_beta{β}_gamma{γ}.png', dpi=300)

    sol = saturated(dist, zb, β, γ, α=α, dealias=dealias)
    fig, ax = plot_solution(sol)
    fig.tight_layout()
    fig.savefig(f'analytic_saturated_alpha{α}_beta{β}_gamma{γ}.png', dpi=300)

    m = (sol['b'] + γ*sol['q']).evaluate()
    m_bot = m(z=0).evaluate()['g'][0,0,0]
    m_top = m(z=1).evaluate()['g'][0,0,0]
    print("saturated atmosphere properties:")
    print("m_bot = {:.2g}, m_top = {:.2g}, Δm = {:.2g}".format(m_bot, m_top, m_top-m_bot))


    from analytic_zc import f_zc as zc_analytic
    from analytic_zc import f_Tc as Tc_analytic
    zc = zc_analytic()
    Tc = Tc_analytic()

    sol = unsaturated(dist, zb, β, γ, zc(γ), Tc(γ), α=α, q0=q0, dealias=dealias)
    mask = (sol['z']['g'] >= zc(γ))
    fig, ax = plot_solution(sol, mask=mask)
    plot_solution(sol, ax=ax, mask=~mask, linestyle='dashed')
    fig.tight_layout()
    fig.savefig(f'analytic_unsaturated_q{q0}_alpha{α}_beta{β}_gamma{γ}.png', dpi=300)

    m = (sol['b'] + γ*sol['q']).evaluate()
    m_bot = m(z=0).evaluate()['g'][0,0,0]
    m_top = m(z=1).evaluate()['g'][0,0,0]
    print("unsaturated atmosphere properties:")
    print("m_bot = {:.2g}, m_top = {:.2g}, Δm = {:.2g}".format(m_bot, m_top, m_top-m_bot))

    fig = None
    alpha_f = lambda β: 0.75*(1.35-β)/.25
    for i, β in enumerate(np.linspace(1,1.25, num=6)):
        sol = saturated(dist, zb, β, γ, α=α, dealias=dealias)
        if fig:
            plot_solution(sol, ax=ax, alpha=alpha_f(β))
        else:
            fig, ax = plot_solution(sol)
    fig.tight_layout()
    fig.savefig(f'many_beta_analytic_saturated_alpha{α}_gamma{γ}.png', dpi=300)

    fig = None
    alpha_f = lambda β: 0.75*(1.2-β)/.15
    for β in np.linspace(1,1.15, num=6):
        sol = unsaturated(dist, zb, β, γ, zc(γ), Tc(γ), α=α, q0=q0, dealias=dealias)
        mask = (sol['z']['g'] >= zc(γ))
        if fig:
            plot_solution(sol, mask=mask, ax=ax, alpha=alpha_f(β))
        else:
            fig, ax = plot_solution(sol, mask=mask)
        plot_solution(sol, ax=ax, mask=~mask, linestyle='dashed', alpha=alpha_f(β))

    fig.tight_layout()
    fig.savefig(f'many_beta_analytic_unsaturated_q{q0}_alpha{α}_gamma{γ}.png', dpi=300)
