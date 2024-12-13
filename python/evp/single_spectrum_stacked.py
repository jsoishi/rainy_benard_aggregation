"""
Dedalus script for plotting spectrum of static drizzle solutions to the Rainy-Benard system of equations.

Read more about these equations in:

Vallis, Parker & Tobias, 2019, JFM,
``A simple system for moist convection: the Rainy–Bénard model''

This script solves EVPs for an existing atmospheres, solved for by scripts in the nlbvp section.

Usage:
    convective_onset.py [options]

Options:
                           Properties of analytic atmosphere
    --alpha=<alpha>        alpha value [default: 3]
    --beta=<beta>          beta value  [default: 1.05]
    --gamma=<gamma>        gamma value [default: 0.19]
    --q0=<q0>              basal q value [default: 0.6]

    --tau=<tau>            If set, override value of tau [default: 1e-3]
    --k=<k>                If set, override value of k [default: 1e5]

    --nondim=<n>           Non-Nondimensionalization [default: buoyancy]

    --Ra=<Ra>              Rayleigh number [default: 2.68e4]
    --kx=<kx>              x wavenumber [default: 2.57]
    --top-stress-free      Stress-free upper boundary
    --stress-free          Stress-free both boundaries

    --nz=<nz>              Number of coeffs to use in eigenvalue search [default: 128]
    --target=<targ>        Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>          Target number of eigenvalues to search for [default: 20]
    --normalization=<norm>      Eigenmode plot Normalization [default: m]

    --erf                  Use an erf rather than a tanh for the phase transition
    --Legendre             Use Legendre polynomials
    --drift_threshold=<dt>        Drift threshold [default: 1e6]
    --rejection_method=<rej>      Method for rejecting modes [default: resolution]

    --dense                Solve densely for all eigenvalues (slow)
    --emode=<emode>        Index for eigenmode to visualize
    --annotate             Print mode indices on plot for identification
    --plot_type=<plot_type>   File type for plots [default: pdf]
    --use-heaviside        Use the Heaviside function
    --restart              Don't solve, use saved eigenmodes and eigenvalues
    --dynamic-gamma=<dyn_gam>       Factor to raise or lower gamma by in EVP (but not background!) [default: 1]
"""
import logging
logger = logging.getLogger(__name__)
for system in ['matplotlib', 'PIL']:
    logging.getLogger(system).setLevel(logging.WARNING)
import numpy as np
import matplotlib.pyplot as plt
try:
    plt.style.use('prl.mplstyle')
except OSError:
    plt.style.use('prl')

from rainy_evp import RainySpectrum
if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)

    N_evals = int(float(args['--eigs']))
    target = float(args['--target'])
    Rayleigh = float(args['--Ra'])
    tau_in = float(args['--tau'])
    drift_threshold = float(args['--drift_threshold'])
    normalization = args['--normalization']
    if args['--stress-free']:
        bc_type = 'stress-free'
    elif args['--top-stress-free']:
        bc_type = 'top-stress-free'
    else:
        bc_type = None # default no-slip
    kx = float(args['--kx'])
    annotate = args['--annotate']
    plot_type = args['--plot_type']
    use_heaviside = args['--use-heaviside']
    restart = args['--restart']
    rejection_method= args['--rejection_method']
    dynamic_gamma = float(args['--dynamic-gamma'])
    if dynamic_gamma != 1.0:
        logger.warning(f"dynamic_gamma factor is set to {dynamic_gamma:}. This is not self-consistent!")
    Prandtlm = 1
    Prandtl = 1
    dealias = 1
    dense = args['--dense']

    emode = args['--emode']
    if emode:
        emode = int(emode)

    Legendre = args['--Legendre']
    erf = args['--erf']
    case = 'analytic' #args['<case>']
    nondim = args['--nondim']
    if case == 'analytic':
        α = float(args['--alpha'])
        β = float(args['--beta'])
        γ = float(args['--gamma'])
        k = float(args['--k'])
        lower_q0 = float(args['--q0'])
        tau = float(args['--tau'])
    else:
        raise NotImplementedError
    if args['--nz']:
        nz = int(float(args['--nz']))
    else:
        nz = nz_sol

    if args['--tau']:
        tau_in = float(args['--tau'])
    # build solvers
    spectrum = RainySpectrum(nz, Rayleigh, tau, kx, γ, α, β, lower_q0, k, Legendre=Legendre, erf=erf, bc_type=bc_type, nondim=nondim, dealias=dealias,Lz=1, use_heaviside=use_heaviside, restart=restart, rejection_method=rejection_method, dynamic_gamma_factor=dynamic_gamma)

    spectrum.lo_res.plot_background()

    spectrum.solve(dense=dense, N_evals=N_evals, target=target, plot_drift_ratios=True)
    print(40*'*')
    spectrum.lo_res.solver.print_subproblem_ranks()
    print(40*'*')

    evals_good = spectrum.evals_good
    indx = spectrum.indx
    fig = plt.figure(figsize=[6,6])
    spec_ax = fig.add_axes([0.15,0.2,0.8,0.7])
    fig_filename=f"Ra_{Rayleigh:.2e}_nz_{nz}_kx_{kx:.3f}_bc_{bc_type}_dense_{dense}_rejection_{rejection_method}_dynamic_gamma_{dynamic_gamma:}_spectrum"
    if restart:
        fig_filename += "_restart"
    logger.info(f"good modes ({{$\delta_t$}} = {drift_threshold:.1e}):    max growth rate = {spectrum.evals_good[-1]}")
    eps = 1e-3
    logger.info(f"good fastest oscillating modes: {spectrum.evals_good[np.argmax(np.abs(spectrum.evals_good.imag))]}")
    col = np.where(np.abs(evals_good.imag) > eps, '#3182bd', np.where(evals_good.real > eps, '#e34a33', np.where(np.abs(evals_good.real) < eps, '#fdbb84', 'k')))
    spec_ax.scatter(evals_good.real, evals_good.imag, marker='o', c=col, label=f'good modes ($\delta_t$ = {drift_threshold:.1e})',s=100)#, alpha=0.5, s=25)
    #col = np.where(np.abs(evals_good.imag) > eps, 'g', np.where(evals_good.real > 0, 'r','k'))

    if annotate:
        for n,ev in enumerate(evals_good):
            logger.info(f"ok ev = {ev}, index = {indx[n]}")
            spec_ax.annotate(indx[n], (ev.real, ev.imag), fontsize=6, ha='left',va='top')

    spec_ax.set_xlabel(r"Growth Rate")
    spec_ax.set_ylabel(r"Frequency")
    fig.tight_layout()
    spec_filename = f'{spectrum.lo_res.case_name}/{fig_filename}_full.{plot_type}'
    logger.info(f"saving full file to {spec_filename}")
    fig.savefig(spec_filename, dpi=300)
    spec_ax.set_xlim(-1,0.2)
    spec_ax.set_ylim(-0.3,0.3)
    spec_ax.axvline(x=0, alpha=0.3, zorder=0, color='xkcd:dark grey')
    spec_filename = f'{spectrum.lo_res.case_name}/{fig_filename}.{plot_type}'
    logger.info(f"saving file to {spec_filename}")
    fig.savefig(spec_filename, dpi=300)
    spec_ax.set_xlim(-0.2,0.5)
    spec_filename = f'{spectrum.lo_res.case_name}/{fig_filename}_zoom.{plot_type}'
    logger.info(f"saving zoomed file to {spec_filename}")
    fig.tight_layout()
    fig.savefig(spec_filename, dpi=300)

    if emode is not None:
        spectrum.lo_res.plot_eigenmode(emode, plot_type=plot_type, normalization=normalization)
    else:
        for i in [-1, -2, -3, -4, -5]:
            emode = indx[i]
            spectrum.lo_res.plot_eigenmode(emode, plot_type=plot_type, normalization=normalization)
