date
python3 run_critical_curve.py --q0=0.6 --gamma=0.19 --nz=128 --tau=1e-2 --k=1e5 --use-heaviside --continuation &
python3 run_critical_curve.py --q0=0.6 --gamma=0.3 --nz=128 --tau=1e-2 --k=1e5 --use-heaviside --continuation &
python3 run_critical_curve.py --q0=0.6 --beta=1 --nz=128 --tau=1e-2 --k=1e5 --use-heaviside --continuation &
python3 run_critical_curve.py --q0=0.6 --beta=1.05 --nz=128 --tau=1e-2 --k=1e5 --use-heaviside --continuation &
python3 run_critical_curve.py --q0=0.6 --beta=1.1 --nz=128 --tau=1e-2 --k=1e5 --use-heaviside --continuation &
wait
date
