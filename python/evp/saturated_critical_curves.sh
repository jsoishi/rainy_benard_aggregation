date
python3 run_critical_curve.py --gamma=0.19 --nz=32 &
python3 run_critical_curve.py --gamma=0.3 --nz=32 &
python3 run_critical_curve.py --beta=1 --nz=32 &
python3 run_critical_curve.py --beta=1.2 --nz=32 &
wait
date
