python3 single_spectrum_stacked.py --top-stress-free --Legendre --erf --use-heaviside --eigs=50 \
                                   --nz=256 --plot=pdf --beta=1.1 --Ra=6.87e5 --kx=2.60 --norm='b'

python3 single_spectrum_stacked.py --top-stress-free --Legendre --erf --use-heaviside --eigs=50 \
                                   --nz=256 --plot=pdf --beta=1.1 --Ra=6.87e6 --kx=6.0

python3 single_spectrum_stacked.py --top-stress-free --Legendre --erf --use-heaviside --eigs=100 \
                                   --nz=256 --plot=pdf --beta=1.1 --Ra=6.87e7 --kx=9.0
