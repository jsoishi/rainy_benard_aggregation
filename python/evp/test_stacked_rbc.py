import numpy as np
import matplotlib.pyplot as plt
from rainy_evp import SplitRainyBenardEVP, RainyBenardEVP, mode_reject
from eigenfunctions import plot_eigenfunctions
plt.style.use('prl')
drift_threshold=1e6
nz = 128
nz_norm = nz
Ra = 1.6e6
tau = 1e-3
alpha = 3
beta = 1.1
gamma = 0.19
kx = 1.34
lower_q0 = 0.6
k = 1e4
sp_lo = SplitRainyBenardEVP(nz, Ra, tau, kx, gamma, alpha, beta, lower_q0, k, Legendre=False)
sp_hi = SplitRainyBenardEVP(2*nz, Ra, tau, kx, gamma, alpha, beta, lower_q0, k, Legendre=False)
norm_lo = RainyBenardEVP(nz_norm, Ra, tau, kx, gamma, alpha, beta, lower_q0, k, Legendre=False)
norm_hi = RainyBenardEVP(2*nz_norm, Ra, tau, kx, gamma, alpha, beta, lower_q0, k, Legendre=False)

sp_lo.plot_background()
for s,n in zip([sp_lo,sp_hi], [norm_lo, norm_hi]):
    s.solve()
    #n.solve()
#norm_evals_good, norm_indx, norm_ep = mode_reject(norm_lo, norm_hi, drift_threshold=drift_threshold)
sp_evals_good, sp_indx, sp_ep = mode_reject(sp_lo, sp_hi, drift_threshold=drift_threshold)


# plt.clf()
# plt.scatter(sp_evals_good.real, sp_evals_good.imag, label='stacked bases')
# plt.scatter(norm_evals_good.real, norm_evals_good.imag,marker='x',label='NCC')
# plt.legend()
# plt.xlabel(r"$\Re{\sigma}$")
# plt.ylabel(r"$\Im{\sigma}$")
# plt.xlim(-10,1)
# plt.ylim(-1,1)
# plt.tight_layout()
# plt.savefig(f"stacked_nz_{nz}_NCC_nz_{nz_norm}_spectra_Ra_{Ra:5.2e}_alpha_{alpha:3.2f}_beta_{beta:3.2f}_gamma_{gamma:3.2f}_kx_{kx:3.2f}_lower_q0_{lower_q0:3.2f}.png",dpi=300)

sp_lo.plot_eigenmode(sp_indx[-1])
#norm_lo.plot_eigenmode(norm_indx[-1])
