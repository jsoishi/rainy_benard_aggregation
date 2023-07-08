import numpy as np

file = 'gamma_0_1_dg_0.1_zc_Tc.csv'
deg = 5

def read_zc(file):
    data = np.genfromtxt(file, delimiter=',', names=True)
    return(data)

def fit_zc(data, deg=deg):
    from numpy.polynomial import Legendre as P
    fit = P.fit(data['gamma'], data['zc'], deg)
    return fit

def f_zc(deg=deg):
    return fit_zc(read_zc(file), deg=deg)

if __name__=='__main__':
    import matplotlib.pyplot as plt

    # continuous fit to data
    zc = f_zc()
    γ = np.linspace(0,1)

    # discrete data
    data = read_zc(file)
    fig, ax = plt.subplots(figsize=[4,4/1.6])
    ax.scatter(data['gamma'], data['zc'], label=r'$z_c$', alpha=0.5)
    ax.plot(γ, zc(γ), alpha=0.5, color='tab:blue')
    ax.scatter(data['gamma'], -data['Tc'], label=r'$-T_c$', color='tab:orange', alpha=0.5)
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$z_c$ and $-T_c$')
    ax.legend()
    fig.tight_layout()
    fig.savefig('analytic_zc.png', dpi=300)

    L2 = np.sum(np.abs(zc(data['gamma'])-data['zc']))
    print('L2 error of fit at nodes: {:.2g}'.format(L2))
