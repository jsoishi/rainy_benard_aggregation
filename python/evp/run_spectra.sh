#!/usr/bin/bash
#SBATCH --partition=faculty
#SBATCH --time=5-0
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --distribution=cyclic:cyclic
#SBATCH --output=slurm_%x_%j.out

export OMPI_MCA_btl=^openib
conda activate d3-local-openmpi4
# beta = 1.1
python3 spectrum.py analytic --tau=0.1 --k=1e4 --nz=128 --q0=0.6 --beta=1.1 --Ra=9.51e11 --erf --Legendre --Lx=4 --dense --top-stress-free &
python3 spectrum.py analytic --tau=0.1 --k=1e4 --nz=128 --q0=0.6 --beta=1.1 --Ra=9.51e10 --erf --Legendre --Lx=4 --dense --top-stress-free & 
python3 spectrum.py analytic --tau=0.1 --k=1e4 --nz=128 --q0=0.6 --beta=1.1 --Ra=9.51e9 --erf --Legendre --Lx=4 --dense --top-stress-free &

#montage analytic_unsaturated/alpha3_beta1.1_gamma0.19/tau0.1_k1e4_erf_Legendre/rainy_benard_Ra9.51e9_tau0.1_k1e+04_nz128_nx512/rh_avg_hov.png analytic_unsaturated/alpha3_beta1.1_gamma0.19/tau0.1_k1e4_erf_Legendre/rainy_benard_Ra9.51e10_tau0.1_k1e+04_nz128_nx512/rh_avg_hov.png analytic_unsaturated/alpha3_beta1.1_gamma0.19/tau0.1_k1e4_erf_Legendre/rainy_benard_Ra9.51e11_tau0.1_k1e+04_nz128_nx512/rh_avg_hov.png -tile 1x -geometry 3900x975+1+1 rh_hov_alpha3_beta1.1_gamma0.19_tau0.1_k1e4_erf_Legendre_S1e4-6.png 

# beta = 1.05
python3 spectrum.py analytic --tau=0.1 --k=1e4 --nz=128 --q0=0.6 --beta=1.05 --Ra=2.56e10 --erf --Legendre --Lx=4 --dense --top-stress-free & 
python3 spectrum.py analytic --tau=0.1 --k=1e4 --nz=128 --q0=0.6 --beta=1.05 --Ra=2.56e9 --erf --Legendre --Lx=4 --dense --top-stress-free & 
python3 spectrum.py analytic --tau=0.1 --k=1e4 --nz=128 --q0=0.6 --beta=1.05 --Ra=2.56e8 --erf --Legendre --Lx=4 --dense --top-stress-free &


# montage analytic_unsaturated/alpha3_beta1.05_gamma0.19/tau0.1_k1e4_erf_Legendre/rainy_benard_Ra2.56e8_tau0.1_k1e+04_nz128_nx512/rh_avg_hov.png analytic_unsaturated/alpha3_beta1.05_gamma0.19/tau0.1_k1e4_erf_Legendre/rainy_benard_Ra2.56e9_tau0.1_k1e+04_nz128_nx512/rh_avg_hov.png analytic_unsaturated/alpha3_beta1.05_gamma0.19/tau0.1_k1e4_erf_Legendre/rainy_benard_Ra2.56e10_tau0.1_k1e+04_nz128_nx512/rh_avg_hov.png -tile 1x -geometry 3900x975+1+1 rh_hov_alpha3_beta1.05_gamma0.19_tau0.1_k1e4_erf_Legendre_S1e4-6.png 

# beta = 1
python3 spectrum.py analytic --tau=0.1 --k=1e4 --nz=128 --q0=0.6 --beta=1. --Ra=1.11e10 --erf --Legendre --Lx=4 --dense --top-stress-free &
python3 spectrum.py analytic --tau=0.1 --k=1e4 --nz=128 --q0=0.6 --beta=1. --Ra=1.11e9 --erf --Legendre --Lx=4 --dense --top-stress-free &
python3 spectrum.py analytic --tau=0.1 --k=1e4 --nz=128 --q0=0.6 --beta=1. --Ra=1.11e8 --erf --Legendre --Lx=4 --dense --top-stress-free &

wait
# montage analytic_unsaturated/alpha3_beta1_gamma0.19/tau0.1_k1e4_erf_Legendre/rainy_benard_Ra1.11e8_tau0.1_k1e+04_nz128_nx512/rh_avg_hov.png analytic_unsaturated/alpha3_beta1_gamma0.19/tau0.1_k1e4_erf_Legendre/rainy_benard_Ra1.11e9_tau0.1_k1e+04_nz128_nx512/rh_avg_hov.png analytic_unsaturated/alpha3_beta1_gamma0.19/tau0.1_k1e4_erf_Legendre/rainy_benard_Ra1.11e10_tau0.1_k1e+04_nz128_nx512/rh_avg_hov.png -tile 1x -geometry 3900x975+1+1 rh_hov_alpha3_beta1_gamma0.19_tau0.1_k1e4_erf_Legendre_S1e4-6.png 
