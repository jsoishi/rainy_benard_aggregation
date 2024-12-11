date
python3 run_critical_curve.py --q0=0.6 --gamma=0.19 --nz=256 --tau=1e-3 --k=1e5 --Ra_guess=2e5 --kx_guess=2.5 --use-heaviside --continuation &
python3 run_critical_curve.py --q0=0.6 --gamma=0.3 --nz=256 --tau=1e-3 --k=1e5 --Ra_guess=2e5 --kx_guess=2.5 --use-heaviside --continuation &
python3 run_critical_curve.py --q0=0.6 --beta=1 --nz=256 --tau=1e-3 --k=1e5 --Ra_guess=2e4 --kx_guess=2.5 --use-heaviside --continuation &
python3 run_critical_curve.py --q0=0.6 --beta=1.05 --nz=256 --tau=1e-3 --k=1e5 --Ra_guess=2e4 --kx_guess=2.5 --use-heaviside --continuation &
python3 run_critical_curve.py --q0=0.6 --beta=1.1 --nz=256 --tau=1e-3 --k=1e5 --Ra_guess=2e4 --kx_guess=2.5 --use-heaviside --continuation &
wait
date
