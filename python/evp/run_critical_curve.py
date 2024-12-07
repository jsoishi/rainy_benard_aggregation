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

     --min_Ra=<mnRa>      Min Ra to search [default: 1e3]
     --max_Ra=<mxRa>      Max Ra to search [default: 1e7]
     --num_Ra=<nRa>       Num Ra to search [default: 2]

     --min_kx=<mnkx>      Min kx to search [default: 1]
     --max_kx=<mxkx>      Max kx to search [default: 3]
     --num_kx=<nkx>       Num kx to search [default: 5]

     --use-heaviside

     --tau=<tau>          Timescale for moisture reaction [default: 0.01]
     --k=<k>              Smooth heaviside width [default: 1e4]

     --nz=<nz>            z resolution [default: 128]
     --verbose

"""
import subprocess as sp
import numpy as np
from docopt import docopt
args = docopt(__doc__)

q0 = float(args['--q0'])
taus = [float(args['--tau'])]
k = args['--k']

min_Ra = float(args['--min_Ra'])
max_Ra = float(args['--max_Ra'])
num_Ra = int(float(args['--num_Ra']))

min_kx = float(args['--min_kx'])
max_kx = float(args['--max_kx'])
num_kx = int(float(args['--num_kx']))

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
    print(gammas)
else:
    raise ValueError("neither gamma nor beta specified; select one to fix")

if args['--use-heaviside']:
    use_H = '--use-heaviside'
else:
    use_H = ''

for γ in gammas:
    for β in betas:
        for tau in taus:
            print(f"solving γ = {γ}, β = {β}:")
            run_command = f"python3 convective_onset.py --beta={β} --gamma={γ} --q0={q0} \
                     --nz={nz} --k={k} --tau={tau} \
                     --min_Ra={min_Ra} --max_Ra={max_Ra} --num_Ra={num_Ra} \
                     --min_kx={min_kx} --max_kx={max_kx} --num_kx={num_kx} \
                     --erf --Legendre --top-stress-free {use_H}"
            if args['--verbose']:
                print(run_command)
            sp.run(run_command, shell=True, capture_output=True)
