"""
Script for automating critical_Ra calculations.

The approach is to either fix gamma and sweep beta, or fix beta and sweep gamma.
This is done by selecting a value of gamma or beta via the keyword arguments.

Usage:
     run_critical_curve.py [options]

Options:
     --gamma=<gamma>      gamma to solve at
     --beta=<beta>        beta to solve at

     --min_Ra=<mnRa>      Min Ra to search [default: 1e3]
     --max_Ra=<mxRa>      Max Ra to search [default: 1e7]
     --num_Ra=<nRa>       Num Ra to search [default: 2]

     --min_kx=<mnkx>      Min kx to search [default: 1]
     --max_kx=<mxkx>      Max kx to search [default: 3]
     --num_kx=<nkx>       Num kx to search [default: 5]

     --nz=<nz>            z resolution [default: 128]

"""
import subprocess as sp
import numpy as np
from docopt import docopt
args = docopt(__doc__)

min_Ra = float(args['--min_Ra'])
max_Ra = float(args['--max_Ra'])
num_Ra = int(float(args['--num_Ra']))

min_kx = float(args['--min_kx'])
max_kx = float(args['--max_kx'])
num_kx = int(float(args['--num_kx']))

nz = int(args['--nz'])

if args['--gamma']:
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
    max_γ = 0.4
    Δγ = 0.025
    num_γ = int(np.round((max_γ - min_γ)/Δγ)+1)
    print(f"sweeping γ = [{min_γ, max_γ}] with a total of {num_γ} samples")
    gammas = np.linspace(min_γ, max_γ, num=num_γ)
    gammas = np.round(gammas, decimals=3)
    print(gammas)
else:
    raise ValueError("neither gamma nor beta specified; select one to fix")

for γ in gammas:
    for β in betas:
        print(f"solving γ = {γ}, β = {β}:")
        sp.run(f"python3 convective_onset.py --beta={β} --gamma={γ} --q0=1 \
                 --nz={nz} --k=1e4 --tau=0.1 \
                 --min_Ra={min_Ra} --max_Ra={max_Ra} --num_Ra={num_Ra} \
                 --min_kx={min_kx} --max_kx={max_kx} --num_kx={num_kx} \
                 --erf --Legendre --top-stress-free",
                 shell=True, capture_output=True)
