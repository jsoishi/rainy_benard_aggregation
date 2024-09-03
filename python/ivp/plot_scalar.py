"""
Plot scalar outputs from scalar_output.h5 file.

Usage:
    plot_scalar.py <file> [options]

Options:
    --times=<times>      Range of times to plot over; pass as a comma separated list with t_min,t_max.  Default is whole timespan.
    --output=<output>    Output directory; if blank, a guess based on <file> location will be made.

    --lombscargle        Do an explicit lombscargle periodogram (we always do a FFT autocorrelation)
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
    data_dir = args['<file>'].split('scalars')[0]
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

energy_keys = ['KE','PE','QE','ME']

fig_E, ax_E = plt.subplots(nrows=2, sharex=True)
for key in energy_keys:
    ax_E[0].plot(t, data[key], label=key)
for key in ['KE', 'ME']:
    ax_E[1].plot(t, data[key]-data[key][0], label=key+"'")

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
    ax_tau[i].plot(t, data['τ_d'], label=r'$\tau_{d}$')
    ax_tau[i].plot(t, data['τ_u'], label=r'$\tau_{u}$')
    ax_tau[i].plot(t, data['τ_b'], label=r'$\tau_{b}$')
    ax_tau[i].plot(t, data['τ_q'], label=r'$\tau_{q}$')

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


fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(t, data['div_u'], label=r'$\vec{\nabla}\cdot\vec{u}$')
ax[1].plot(t, np.abs(data['div_u']), label=r'$|\vec{\nabla}\cdot\vec{u}|$')

for a in ax:
    if subrange:
        a.set_xlim(t_min,t_max)
    a.set_xlabel('time')
    a.legend()
ax[1].set_yscale('log')
fig.tight_layout()
fig.savefig('{:s}/div_u_error.png'.format(str(output_path)), dpi=300)


fig_f, ax_f = plt.subplots(nrows=2, sharex=True)
for ax in ax_f:
    ax.plot(t, data['Re'], label='Re')
    ax_r = ax.twinx()
    ax_r.plot(t, data['enstrophy'], label=r'$\omega^2$', color='tab:orange')
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel('Re')
    ax_r.set_ylabel(r"$\omega^2$")

handles_re, labels_re = ax.get_legend_handles_labels()
handles_ens, labels_ens = ax_r.get_legend_handles_labels()
handles = handles_re + handles_ens
labels = labels_re + labels_ens
ax.legend(handles,labels)

ax_f[1].set_yscale('log')
ax_r.set_yscale('log') # relies on it being the last instance; poor practice

fig_f.tight_layout()
fig_f.savefig('{:s}/Re_and_ens.pdf'.format(str(output_path)))
fig_f.savefig('{:s}/Re_and_ens.png'.format(str(output_path)), dpi=300)

benchmark_set = ['KE', 'PE', 'QE', 'Re', 'enstrophy', 'τu1', 'τu2', 'τb1', 'τb2', 'τq1', 'τq2', 'τp', 'div_u']

i_ten = int(0.9*data[benchmark_set[0]].shape[0])
print("total simulation time {:6.2g}".format(t[-1]-t[0]))
print("benchmark values (averaged from {:g}-{:g})".format(t[i_ten], t[-1]))
for benchmark in benchmark_set:
    try:
        print("{:3s} = {:20.12e} +- {:4.2e}".format(benchmark, np.mean(data[benchmark][i_ten:]), np.std(data[benchmark][i_ten:])))
    except:
        print("{:3s} missing".format(benchmark))

def Nuttal_window(x_in):
    # implementing our own window to handle non-uniform grid spacing
    # https://en.wikipedia.org/wiki/Window_function
    a_0=0.355768
    a_1=0.487396
    a_2=0.144232
    a_3=0.012604
    x = x_in - np.min(x_in)
    Δx = np.max(x)-np.min(x)
    return a_0 - a_1*np.cos(2*np.pi*x/Δx) + a_2*np.cos(4*np.pi*x/Δx) - a_3*np.cos(6*np.pi*x/Δx)

print("periodogram analysis")
fig, ax = plt.subplots(figsize=[6,6],nrows=2)

if not subrange:
    t_min = 0.2*np.max(t-np.min(t))+np.min(t)
    t_max = np.max(t)
mask = ((t>=t_min) & (t<=t_max))
print(f'analyzing time subrange: {t_min:.2g}--{t_max:.2g}')
ts = t[mask]
f_min = 2*2*np.pi/(np.max(ts)-np.min(ts))
f_max = 1e2*f_min
print(f'f_min: {f_min:6.2g}, P(f_min): {2*np.pi/f_min:6.2g}')
print(f'f_max: {f_max:6.2g}, P(f_max): {2*np.pi/f_max:6.2g}')
N_freq = int(1e4)
print(f'sampling with N = {N_freq} log-sampled freqs')



print("  quantity | freq (period)")
print("--------------------------")
import scipy.signal as scs
for q in ['KE', 'PE', 'QE', 'Re', 'enstrophy']:
    ds = np.copy(data[q][mask])
    ds -= np.mean(ds)
    ds /= np.std(ds)
    ds *= Nuttal_window(ts)
    if args['--lombscargle']:
        freqs = np.geomspace(f_min, f_max, N_freq)
        power = scs.lombscargle(ts, ds, freqs, normalize=True, precenter=True)
        freqs /= 2*np.pi # same approach as FFT frequencies
    else:
        coeffs = np.fft.rfft(ds, norm='ortho')
        power = (coeffs*np.conj(coeffs)).real
        n = ds.size
        dt = np.mean(ts[1:-1]-ts[0:-2])
        freqs = np.fft.rfftfreq(n, d=1/dt)

    i_max = np.argmax(power)
    print("{:>10s} = {:.3g} ({:.3g})".format(q, freqs[i_max], 1/freqs[i_max]))

    ax[0].plot(freqs, power, alpha=0.5)
    ax[0].scatter(freqs[i_max], power[i_max], marker='o', label=f'{q:s} = {1/freqs[i_max]:.3g}')
    for i in range(3):
        ax[0].axvline(x=(i+1)*freqs[i_max], linestyle='dashed', color='xkcd:dark grey', alpha=0.1)
    ax[-1].plot(ts, ds, label=q)
ax[0].legend(framealpha=0.2, fontsize=8)
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_ylabel('periodogram')
ax[0].set_xlabel('frequency')
if args['--lombscargle']:
    ax[0].set_ylim(1e-4,1)
ax[-1].set_ylabel('scaled quantity')
ax[-1].set_xlabel('time')
fig.tight_layout()
fig.savefig('{:s}/periodogram.png'.format(str(output_path)), dpi=300)
