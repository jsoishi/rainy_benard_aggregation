date
python3 run_critical_curve.py --gamma=0.19 --nz=128 --tau=1e-2 --min_Ra=2e3 --max_Ra=2e5 --num_Ra=11 --min_kx=1 --max_kx=5 --num_kx=11 &
python3 run_critical_curve.py --gamma=0.3 --nz=128 --tau=1e-2 --min_Ra=2e3 --max_Ra=2e5 --num_Ra=11 --min_kx=1 --max_kx=5 --num_kx=11 &
python3 run_critical_curve.py --beta=1 --nz=128 --tau=1e-2 --min_Ra=2e3 --max_Ra=2e5 --num_Ra=11 --min_kx=1 --max_kx=5 --num_kx=11 &
python3 run_critical_curve.py --beta=1.2 --nz=128 --tau=1e-2 --min_Ra=2e3 --max_Ra=2e5 --num_Ra=11 --min_kx=1 --max_kx=5 --num_kx=11 &
wait
date
