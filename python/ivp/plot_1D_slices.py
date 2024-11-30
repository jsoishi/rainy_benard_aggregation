"""
Plot horizontally extracted profiles (at fixed z).

Usage:
    plot_1D_slices.py <file> [options]

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

def plot_hov(df, task_name, output_path, zrange=None, aspect_ratio = 4, fig_W = 13, t_avg_start=1000, eps=0.05):
    fig_H = fig_W/aspect_ratio
    fig = plt.figure(figsize=[fig_W,fig_H])
    hov_ax = fig.add_axes([0.07,0.2,0.8,0.6])
    hov_cb_ax = fig.add_axes([0.07,0.8,0.2,0.04])
    plot_z_ax = fig.add_axes([0.87,0.2,0.1,0.6])

    print(task_name)
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

    xx,yy = np.meshgrid(t, x)

    # zrange sets the range for colorbars and inset plot
    if zrange:
        zmin, zmax = zrange
    else:
        zmin = zmax = None

    hov_img = hov_ax.pcolormesh(xx, yy, task[(slice(None),*mask)].T, rasterized=True, vmin=zmin, vmax=zmax)
    hov_ax.set_xlabel(r"$t$")
    hov_ax.set_ylabel(r"$x$")
    cb = fig.colorbar(hov_img, orientation='horizontal',cax=hov_cb_ax)
    cb.set_label(label=f"{task_label}",fontsize=14)
    hov_cb_ax.xaxis.tick_top()
    hov_cb_ax.xaxis.set_label_position('top')
    hov_cb_ax.tick_params(axis='x', which='major',labelsize=10, pad=2)
    if t[-1] > 1.5*t_avg_start:
        i_avg_start = np.argmin(np.abs(t-t_avg_start))
    else:
        i_avg_start = int(len(t)/2)
    task_tavg = task[(slice(i_avg_start,None),*mask)].mean(axis=0)
    task_ic = task[(0,*mask)]
    plot_z_ax.plot(task_tavg,x)
    plot_z_ax.plot(task_ic, x)
    plot_z_ax.set_ylim(hov_ax.get_ylim())
    if zmin is not None:
        if zmax == 1:
            zmax += eps
        # here the x-direction is the line amplitude, hence zrange
        plot_z_ax.set_xlim(zmin, zmax)
    plot_z_ax.get_yaxis().set_visible(False)
    plot_z_ax.set_xlabel(f"{task_label}")
    plot_z_ax.xaxis.tick_top()
    plot_z_ax.xaxis.set_label_position('top')
    fig.savefig(output_path/pathlib.Path(f"{task_name}_hov_x.png"), dpi=300)

args = docopt(__doc__)
df_name = args['<file>']
case = df_name.split('slice')[0]
if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = case +'/'
    output_path = pathlib.Path(data_dir).absolute()

tasks = {'b':None, 'q':None, 'm':None, 'rh':(0,1), 'ub':None, 'uq':None, 'uz':None, 'omega_y':None}

with h5py.File(df_name,'r') as df:
    for task, zr in tasks.items():
        plot_hov(df, task, output_path, zrange=zr)
