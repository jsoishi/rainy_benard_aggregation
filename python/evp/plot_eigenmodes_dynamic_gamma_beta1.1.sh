python3 single_spectrum.py --top-stress-free --Legendre --erf --use-heaviside --eigs=50 \
                           --nz=256 --plot=pdf --beta=1.1 --Ra=6.87e5 --kx=2.60 \
                           --norm='b' --annotate
# eigenmode 6 is the one used in the paper                           

python3 single_spectrum.py --top-stress-free --Legendre --erf --use-heaviside --eigs=50 \
                          --nz=256 --plot=pdf --beta=1.1 --Ra=6.87e5 --kx=2.60 \
                          --norm='b' --annotate \
                          --dynamic_gamma=0
