"""
Script for automating critical_Ra calculations.

The approach is to either fix gamma and sweep beta, or fix beta and sweep gamma.
This is done by selecting a value of gamma or beta via the keyword arguments.
If both beta and gamma are selected, than instead tau is swept.

Usage:
     run_critical_curve.py [options]

Options:
     --gamma=<gamma>      gamma to solve at
     --beta=<beta>        beta to solve at
     --q0=<q0>            Lower atmosphere moisture [default: 1]

     --grid-search     Use a grid based search, which takes the following options
     --min_Ra=<mnRa>      Min Ra to search [default: 1e3]
     --max_Ra=<mxRa>      Max Ra to search [default: 1e7]
     --num_Ra=<nRa>       Num Ra to search [default: 2]

     --min_kx=<mnkx>      Min kx to search [default: 1]
     --max_kx=<mxkx>      Max kx to search [default: 3]
     --num_kx=<nkx>       Num kx to search [default: 5]

     --Ra_guess=<Rag>  Starting point for Ra search [default: 1e4]
     --kx_guess=<kxg>  Starting point for kx search [default: 2.5]

     --continuation    Use very simple continuation approach

     --use-heaviside

     --tau=<tau>          Timescale for moisture reaction [default: 0.01]
     --k=<k>              Smooth heaviside width [default: 1e4]

     --nz=<nz>            z resolution [default: 128]
     --verbose

"""
import logging
logger = logging.getLogger(__name__)
for system in ['h5py']:
    logging.getLogger(system).setLevel(logging.WARNING)

import subprocess as sp
import numpy as np
from docopt import docopt
args = docopt(__doc__)

q0 = float(args['--q0'])
taus = [float(args['--tau'])]
k = args['--k']

ΔT = -1
α = 3
γ_crit = lambda β: -(β+ΔT)/(np.exp(α*ΔT)-q0)
β_crit = lambda γ: -γ*(np.exp(α*ΔT)-q0) - ΔT

min_Ra = float(args['--min_Ra'])
max_Ra = float(args['--max_Ra'])
num_Ra = int(float(args['--num_Ra']))

min_kx = float(args['--min_kx'])
max_kx = float(args['--max_kx'])
num_kx = int(float(args['--num_kx']))

grid_search = args['--grid-search']
Ra_guess = float(args['--Ra_guess'])
kx_guess = float(args['--kx_guess'])

nz = int(args['--nz'])

if args['--gamma'] and args['--beta']:
    gammas = [float(args['--gamma'])]
    betas = [float(args['--beta'])]
    min_tau = 1e-3
    max_tau = 1e-1
    num_tau = 5
    taus = np.geomspace(min_tau,max_tau, num=num_tau)
    print(f"sweeping tau = [{min_tau, max_tau}] with a total of {num_tau} samples")
elif args['--gamma']:
    gammas = [float(args['--gamma'])]

    min_β = 1.0
    max_β = 1.3
    Δβ = 0.0125
    num_β = int(np.round((max_β - min_β)/Δβ)+1)
    print(f"sweeping β = [{min_β, max_β}] with a total of {num_β} samples")
    betas = np.linspace(min_β, max_β, num=num_β)
    betas = np.round(betas, decimals=4)
    print(betas)
elif args['--beta']:
    betas = [float(args['--beta'])]

    min_γ = 0.025
    max_γ = 0.6
    Δγ = 0.025
    num_γ = int(np.round((max_γ - min_γ)/Δγ)+1)
    print(f"sweeping γ = [{min_γ, max_γ}] with a total of {num_γ} samples")
    gammas = np.linspace(min_γ, max_γ, num=num_γ)
    gammas = np.round(gammas, decimals=3)
    gammas = np.flip(gammas)
    print(gammas)
else:
    raise ValueError("neither gamma nor beta specified; select one to fix")

if args['--use-heaviside']:
    use_H = '--use-heaviside'
else:
    use_H = ''

continuation = args['--continuation']
for γ in gammas:
    for β in betas:
        if β > β_crit(γ) or γ < γ_crit(β):
            print(f"β {β} > β_crit(γ) {β_crit(γ):.3g} or γ {γ} < γ_crit(β) {γ_crit(β):.3g}, breaking")
            continue
        for tau in taus:
            if args['--grid-search']:
                print(f"solving γ = {γ}, β = {β}:")
            else:
                print(f"solving γ = {γ}, β = {β} with guess {kx_guess:.4g}, {Ra_guess:.4g}:")
            run_command = f"python3 convective_onset.py --beta={β} --gamma={γ} --q0={q0} \
                     --nz={nz} --k={k} --tau={tau} \
                     --erf --Legendre --top-stress-free {use_H}"
            if args['--grid-search']:
                run_command += ' --grid-search'
                run_command += f' --min_Ra={min_Ra} --max_Ra={max_Ra} --num_Ra={num_Ra}'
                run_command += f' --min_kx={min_kx} --max_kx={max_kx} --num_kx={num_kx}'
            else:
                run_command += f'--Ra_guess={Ra_guess} --kx_guess={kx_guess} --log-search'
            if args['--verbose']:
                print(run_command)
            result = sp.run(run_command, shell=True, capture_output=True, text=True)
            if continuation:
                for line in result.stdout.splitlines():
                    if 'Critical Ra=' in line:
                        Ra_guess = float(line.split('Critical Ra=')[-1])
                    if 'Critical kx=' in line:
                        kx_guess = np.abs(float(line.split('Critical kx=')[-1]))
            if kx_guess < 2:
                print(f"suspect incorrect minimum, resetting guesses for next run")
                kx_guess = float(args['--kx_guess'])
                Ra_guess = float(args['--Ra_guess'])
