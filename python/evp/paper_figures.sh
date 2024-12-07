# stability wedge diagrams
python3 ideal_stability.py saturated --zoom --one
python3 ideal_stability.py unsaturated --zoom --one

# Critical Ra figures
python3 plot_critical_Ra.py analytic_saturated/*/*/critical_curves_nz_32.h5 --overlay_VPT19
python3 plot_critical_Ra.py analytic_unsaturated/*/*/critical_curves_nz_32.h5 --q0=0.6
