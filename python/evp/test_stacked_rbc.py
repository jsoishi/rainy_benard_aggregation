import numpy as np
import matplotlib.pyplot as plt
from rainy_evp import SplitRainyBenardEVP, RainyBenardEVP
plt.style.use('prl')
nz = 32
Ra = 1e6
tau = 1e-3
alpha = 3
beta = 1.2
gamma = 0.3
kx = 1.34
lower_q0 = 0.6
k = 1e4
sp = SplitRainyBenardEVP(nz, Ra, tau, kx, gamma, alpha, beta, lower_q0, k, Legendre=False)
norm = RainyBenardEVP(nz, Ra, tau, kx, gamma, alpha, beta, lower_q0, k, Legendre=False)
sp.plot_background()

norm.solve()
sp.solve()
plt.clf()
plt.scatter(sp.eigenvalues.real, sp.eigenvalues.imag)
plt.xlabel(r"$\Re{\sigma}$")
plt.ylabel(r"$\Im{\sigma}$")
plt.tight_layout()
plt.savefig("test_spectra.png",dpi=300)
