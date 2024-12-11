date
python3 run_critical_curve.py --gamma=0.19 --nz=128 --tau=1e-3 --k=1e5 --continuation &
python3 run_critical_curve.py --gamma=0.3 --nz=128 --tau=1e-3 --k=1e5 --continuation &
python3 run_critical_curve.py --beta=1 --nz=128 --tau=1e-3 --k=1e5 --continuation &
python3 run_critical_curve.py --beta=1.2 --nz=128 --tau=1e-3 --k=1e5 --continuation &
wait
date
