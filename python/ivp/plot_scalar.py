"""
Plot scalar outputs from scalar_output.h5 file.

Usage:
    plot_scalar.py <file> [options]

Options:
    --times=<times>      Range of times to plot over; pass as a comma separated list with t_min,t_max.  Default is whole timespan.
    --output=<output>    Output directory; if blank, a guess based on <file> location will be made.
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from docopt import docopt
args = docopt(__doc__)
file = args['<file>']

if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = args['<file>'].split('traces')[0]
    data_dir += '/'
    output_path = pathlib.Path(data_dir).absolute()

f = h5py.File(file, 'r')
data = {}
t = f['scales/sim_time'][:]
data_slice = (slice(None),0,0,0)
for key in f['tasks']:
    data[key] = f['tasks/'+key][data_slice]
f.close()

if args['--times']:
    subrange = True
    t_min, t_max = args['--times'].split(',')
    t_min = float(t_min)
    t_max = float(t_max)
    print("plotting over range {:g}--{:g}, data range {:g}--{:g}".format(t_min, t_max, min(t), max(t)))
else:
    subrange = False

energy_keys = ['KE','PE','QE']

fig_E, ax_E = plt.subplots(nrows=2, sharex=True)
for key in energy_keys:
    ax_E[0].plot(t, data[key], label=key)
for key in energy_keys[::2]: # skip PE
    ax_E[1].plot(t, data[key], label=key)

for ax in ax_E:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel('energy density')
    ax.legend(loc='lower left')
fig_E.tight_layout()
fig_E.savefig('{:s}/energies.pdf'.format(str(output_path)))
fig_E.savefig('{:s}/energies.png'.format(str(output_path)), dpi=300)
for ax in ax_E:
    ax.set_yscale('log')
fig_E.savefig('{:s}/log_energies.pdf'.format(str(output_path)))
fig_E.savefig('{:s}/log_energies.png'.format(str(output_path)), dpi=300)

fig_E, ax_E = plt.subplots(nrows=2, sharex=True)
for key in energy_keys:
    ax_E[0].plot(t, data[key]-data[key][0], label=key+"'")
    ax_E[1].plot(t, data[key]-data[key][0], label=key+"'")
ax_E[1].set_yscale('log')
for ax in ax_E:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.legend(loc='lower left')
    ax.set_ylabel('fluc energy density')
fig_E.tight_layout()
fig_E.savefig('{:s}/fluc_energies.pdf'.format(str(output_path)))
fig_E.savefig('{:s}/fluc_energies.png'.format(str(output_path)), dpi=300)



fig_tau, ax_tau = plt.subplots(nrows=2, sharex=True)
for i in range(2):
    ax_tau[i].plot(t, data['τu1'], label=r'$\tau_{u,1}$')
    ax_tau[i].plot(t, data['τu2'], label=r'$\tau_{u,2}$')
    ax_tau[i].plot(t, data['τb1'], label=r'$\tau_{b,1}$')
    ax_tau[i].plot(t, data['τb2'], label=r'$\tau_{b,2}$')
    ax_tau[i].plot(t, data['τq1'], label=r'$\tau_{q,1}$')
    ax_tau[i].plot(t, data['τq2'], label=r'$\tau_{q,2}$')
    ax_tau[i].plot(t, data['τp'], label=r'$\tau_{p}$')

for ax in ax_tau:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel(r'$<\tau>$')
    ax.legend(loc='lower left')
ax_tau[1].set_yscale('log')
ylims = ax_tau[1].get_ylim()
ax_tau[1].set_ylim(max(1e-14, ylims[0]), ylims[1])
fig_tau.tight_layout()
fig_tau.savefig('{:s}/tau_error.pdf'.format(str(output_path)))
fig_tau.savefig('{:s}/tau_error.png'.format(str(output_path)), dpi=300)


fig_f, ax_f = plt.subplots(nrows=2, sharex=True)
for ax in ax_f:
    ax.plot(t, data['Re'], label='Re')
    ax_r = ax.twinx()
    ax_r.plot(t, data['enstrophy'], label=r'$\omega^2$', color='tab:orange')
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel('fluid parameters')
    ax.legend(loc='lower left')

ax_f[1].set_yscale('log')
ax_r.set_yscale('log') # relies on it being the last instance; poor practice

fig_f.tight_layout()
fig_f.savefig('{:s}/Re_and_ens.pdf'.format(str(output_path)))
fig_f.savefig('{:s}/Re_and_ens.png'.format(str(output_path)), dpi=300)

benchmark_set = ['KE', 'PE', 'QE', 'Re', 'enstrophy', 'τu1', 'τu2', 'τb1', 'τb2', 'τq1', 'τq2', 'τp']

i_ten = int(0.9*data[benchmark_set[0]].shape[0])
print("total simulation time {:6.2g}".format(t[-1]-t[0]))
print("benchmark values (averaged from {:g}-{:g})".format(t[i_ten], t[-1]))
for benchmark in benchmark_set:
    try:
        print("{:3s} = {:20.12e} +- {:4.2e}".format(benchmark, np.mean(data[benchmark][i_ten:]), np.std(data[benchmark][i_ten:])))
    except:
        print("{:3s} missing".format(benchmark))
