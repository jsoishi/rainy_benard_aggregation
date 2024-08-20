date
python3 run_critical_curve.py --q0=0.6 --gamma=0.19 --nz=64 --max_Ra=1e6 --num_Ra=7 --max_kx=5 --use-heaviside &
python3 run_critical_curve.py --q0=0.6 --gamma=0.3 --nz=64 --max_Ra=1e6 --num_Ra=7 --max_kx=5 --use-heaviside &
python3 run_critical_curve.py --q0=0.6 --beta=1 --nz=64 --max_Ra=1e6 --num_Ra=7 --max_kx=5 --use-heaviside &
python3 run_critical_curve.py --q0=0.6 --beta=1.05 --nz=64 --max_Ra=1e6 --num_Ra=7 --max_kx=5 --use-heaviside &
python3 run_critical_curve.py --q0=0.6 --beta=1.1 --nz=64 --max_Ra=1e6 --num_Ra=7 --max_kx=5 --use-heaviside &
wait
date
