date
python3 run_critical_curve.py --gamma=0.19 --beta=1.1 --nz=128 --k=1e5 &
python3 run_critical_curve.py --gamma=0.19 --beta=1.175 --nz=128 --k=1e5 --Ra_guess=1e5 &
wait
date
