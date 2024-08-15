python3 plot_L2.py unsaturated_atm_alpha3_beta1.1_gamma0.19_q1.0/tau_* --q0=1
mv unsaturated_atm_alpha3_beta1.1_gamma0.19_q1.0/L2_q.png L2_q_saturated.png
python3 plot_L2.py unsaturated_atm_alpha3_beta1.1_gamma0.19_q0.6/tau_* --q0=0.6
mv unsaturated_atm_alpha3_beta1.1_gamma0.19_q0.6/L2_q.png L2_q_unsaturated.png
