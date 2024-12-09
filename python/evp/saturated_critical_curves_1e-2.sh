date
python3 run_critical_curve.py --gamma=0.19 --nz=64 --tau=1e-2 --k=1e5 --continuation &
python3 run_critical_curve.py --gamma=0.3 --nz=64 --tau=1e-2 --k=1e5 --continuation &
python3 run_critical_curve.py --beta=1 --nz=64 --tau=1e-2 --k=1e5 --continuation &
python3 run_critical_curve.py --beta=1.2 --nz=64 --tau=1e-2 --k=1e5 --continuation &
wait
date
