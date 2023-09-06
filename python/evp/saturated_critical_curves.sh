export gamma=0.19

python3 convective_onset.py --beta=1.175 --gamma=$gamma --q0=1 --nz=128 --k=1e4 --tau=0.1 --min_Ra=1e4 --max_Ra=1e7 --num_Ra=2 --erf --Legendre --top-stress-free

python3 convective_onset.py --beta=1.1625 --gamma=$gamma --q0=1 --nz=128 --k=1e4 --tau=0.1 --min_Ra=1e4 --max_Ra=1e6 --num_Ra=2 --erf --Legendre --top-stress-free

python3 convective_onset.py --beta=1.15 --gamma=$gamma --q0=1 --nz=128 --k=1e4 --tau=0.1 --min_Ra=1e4 --max_Ra=1e6 --num_Ra=2 --erf --Legendre --top-stress-free

python3 convective_onset.py --beta=1.1375 --gamma=$gamma --q0=1 --nz=128 --k=1e4 --tau=0.1 --min_Ra=1e3 --max_Ra=1e6 --num_Ra=2 --erf --Legendre --top-stress-free

python3 convective_onset.py --beta=1.1125 --gamma=$gamma --q0=1 --nz=128 --k=1e4 --tau=0.1 --min_Ra=1e3 --max_Ra=1e5 --num_Ra=2 --erf --Legendre --top-stress-free

python3 convective_onset.py --beta=1.1 --gamma=$gamma --q0=1 --nz=128 --k=1e4 --tau=0.1 --min_Ra=1e3 --max_Ra=1e5 --num_Ra=2 --erf --Legendre --top-stress-free
