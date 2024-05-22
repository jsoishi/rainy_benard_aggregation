date
python3 run_critical_curve.py --q0=0.6 --gamma=0.19 --nz=32 --max_Ra=1e6 --max_kx=5 &
python3 run_critical_curve.py --q0=0.6 --gamma=0.3 --nz=32 --max_Ra=1e6 --max_kx=5 &
python3 run_critical_curve.py --q0=0.6 --beta=1 --nz=32 --max_Ra=1e6 --max_kx=5 &
python3 run_critical_curve.py --q0=0.6 --beta=1.1 --nz=32 --max_Ra=1e6 --max_kx=5 &
wait
date
