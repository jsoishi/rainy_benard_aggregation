
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

NLBVP_sol = {'b':b.copy(), 'q':q.copy(),
             'm':(b+γ*q).evaluate().copy(), 'T':temp.evaluate().copy(), 'rh':rh.evaluate().copy(),
             'tau':tau['g'][0,0,0], 'k':k['g'][0,0,0]}

zc = find_zc(NLBVP_sol)
NLBVP_sol['zc'] = zc
logger.info('tau = {:.1g}, k = {:.0g}, zc = {:.2g}'.format(tau['g'][0,0,0], k['g'][0,0,0], zc))

value = rh.evaluate()
value.change_scales(1)
mask = (value['g'] >= 1-0.01)
ax = plot_solution(NLBVP_sol, title='NLBVP solution', mask=mask, linestyle='solid')
mask = (value['g'] < 1-0.01)
plot_solution(NLBVP_sol, title='NLBVP solution', mask=mask, linestyle='dashed', ax=ax)
print('zc = {:.3g}'.format(NLBVP_sol['zc']))
print('zc = {:.3g}'.format(find_zc(NLBVP_sol)))


plt.show()
