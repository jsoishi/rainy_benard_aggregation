"""
Plot omega-k diagrams (at fixed z).

Adopts form:

     f(x,t) = exp(i kx x + i omega t)

Note that this is shifted 2pi from the Numpy fft.fftfreq() convention.

Usage:
    plot_omega-k.py <file> [options]

Options:
    --output=<output>    Output directory; if blank a guess based on likely case name will be made

"""
import logging
logger = logging.getLogger(__name__.split('.')[-1])

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py

from docopt import docopt
plt.style.use('../../prl.mplstyle')

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

def plot_omega_k(df, task_name, output_path, zrange=None, aspect_ratio = 1.6, fig_W = 13, t_start_frac=0.2, eps=0.05):
    fig_H = fig_W/aspect_ratio
    fig = plt.figure(figsize=[fig_W,fig_H])
    ax = fig.add_axes([0.07,0.1,0.85,0.8])
    cb_ax = fig.add_axes([0.07,0.9,0.2,0.04])

    task_label = task_name.replace("_"," ")
    task = df['tasks'][task_name]
    t = df['scales/sim_time'][:]
    x = task.dims[1][0][:]
    y = task.dims[2][0][:]
    mask = (slice(None),0,0)
    if x.size == 1:
        # half-dimension y is first index, x is second index
        x = y
        mask = (0,slice(None),0)

    data = task[(slice(None),*mask)]

    # trim data and time
    i_t = int(t_start_frac*t.size)
    t = t[i_t:]
    data = data[i_t:,:]

    data *= np.expand_dims(Nuttal_window(t), axis=1)
    coeff = np.fft.rfft2(data)
    coeff = np.fft.fftshift(coeff, axes=0)
    power = (coeff*np.conj(coeff)).real


    delta_x = np.mean(x[1:-1]-x[0:-2])
    Lx = x[-1] - x[0]
    delta_t = np.mean(t[1:-1]-t[0:-2])
    # adopt convention: f(x,t) = exp(i kx x + i omega t), shift fftfreq() by 2pi
    kx = 2*np.pi*np.fft.rfftfreq(x.size, d=delta_x)
    omega = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(t.size, d=delta_t))
    xx,yy = np.meshgrid(kx, omega)
    power = np.log10(power)

    # use a CDF to find the scaling for the plot
    # clip out zero spatial mode, only sample a portion of the lower kx
    kmax = 30
    i_kmax = np.argmin(np.abs(kx-kmax))
    i_ωmin = np.argmin(np.abs(omega-0.5*np.max(omega)))
    i_ωmax = np.argmin(np.abs(omega-0.5*np.min(omega)))
    H, X = np.histogram(power[i_ωmax:i_ωmin,1:i_kmax], bins = 1000, density = True )
    dX = X[1] - X[0]
    F = np.cumsum(H)*dX
    i_min = np.argmin(np.abs(F-0.20))
    i_max = np.argmin(np.abs(F-0.99))
    zmin = X[i_min]
    zmax = X[i_max]

    mask = (power > zmin)
    c_good = coeff
    c_good[~mask] = 0
    phase = np.arctan2(c_good.imag, c_good.real)
    #mask = (power > zmin)
    #phase[~mask] = 0

    print(f'{"log10("+task_name+"):":18s} {np.min(power):6.3g}, {np.std(power):6.3g}, {np.max(power):6.3g}.  Scaling from {zmin:.3g}:{zmax:.3g}')
    hov_img = ax.pcolormesh(xx,yy,power, rasterized=True, vmin=zmin, vmax=zmax)
    ax.set_ylabel(r"$\omega$")
    ax.set_xlabel(r"$k_x$")
    # xlims = ax.get_xlim()
    # ylims = ax.get_ylim()
    ax.set_xlim(0,kmax)
    cb = fig.colorbar(hov_img, orientation='horizontal',cax=cb_ax)
    cb.set_label(label=f"{task_label}",fontsize=14)
    cb_ax.xaxis.tick_top()
    cb_ax.xaxis.set_label_position('top')
    cb_ax.tick_params(axis='x', which='major',labelsize=10, pad=2)

    fig.savefig(output_path/pathlib.Path(f"{task_name}_omega-k.png"), dpi=300)

    fig = plt.figure(figsize=[fig_W,fig_H])
    ax = fig.add_axes([0.07,0.1,0.85,0.8])
    cb_ax = fig.add_axes([0.07,0.9,0.2,0.04])
    hov_img = ax.pcolormesh(xx,yy,phase, rasterized=True, cmap='twilight_shifted')
    ax.set_ylabel(r"$\omega$")
    ax.set_xlabel(r"$k_x$")
    ax.set_xlim(0,kmax)
    cb = fig.colorbar(hov_img, orientation='horizontal',cax=cb_ax)
    cb.set_label(label=f"{task_label}",fontsize=14)
    cb_ax.xaxis.tick_top()
    cb_ax.xaxis.set_label_position('top')
    cb_ax.tick_params(axis='x', which='major',labelsize=10, pad=2)

    fig.savefig(output_path/pathlib.Path(f"{task_name}_omega-k_phase.png"), dpi=300)


args = docopt(__doc__)
df_name = args['<file>']
case = df_name.split('slice')[0]
if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = case +'/'
    output_path = pathlib.Path(data_dir).absolute()

tasks = {'b':None, 'q':None, 'm':None, 'rh':(0,1), 'ub':None, 'uq':None, 'uz':None, 'ux':None, 'omega_y':None, 'enstrophy':None}

with h5py.File(df_name,'r') as df:
    for task, zr in tasks.items():
        plot_omega_k(df, task, output_path, zrange=zr)
