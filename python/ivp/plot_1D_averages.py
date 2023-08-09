"""
Plot horizontally averaged profiles.

Usage:
    plot_1D_avgerages.py <file> [options]

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

def plot_hov(df, task_name, output_path, aspect_ratio = 4, fig_W = 13, t_avg_start=1000):
    fig_H = fig_W/aspect_ratio
    fig = plt.figure(figsize=[fig_W,fig_H])
    hov_ax = fig.add_axes([0.1,0.2,0.8,0.6])
    hov_cb_ax = fig.add_axes([0.1,0.8,0.2,0.04])
    plot_z_ax = fig.add_axes([0.8,0.2,0.1,0.6])

    task_label = task_name.replace("_"," ")
    task = df['tasks'][task_name]
    t = df['scales/sim_time'][:]
    z = task.dims[3][0][:]
    xx,yy = np.meshgrid(t, z)
    print(f"task {task} (min,max) = {task[:].min(), task[:].max()}")
    hov_img = hov_ax.pcolormesh(xx, yy, task[:-1,0,0,:-1].T, rasterized=True)
    hov_ax.set_xlabel(r"$t$")
    hov_ax.set_ylabel(r"$z$")
    cb = fig.colorbar(hov_img, orientation='horizontal',cax=hov_cb_ax)
    cb.set_label(label=f"{task_label}",fontsize=14)
    hov_cb_ax.xaxis.tick_top()
    hov_cb_ax.xaxis.set_label_position('top')
    hov_cb_ax.tick_params(axis='x', which='major',labelsize=10, pad=2)
    task_tavg = task[t_avg_start:,0,0,0:].mean(axis=0)
    task_ic = task[0,0,0,0:]
    plot_z_ax.plot(task_tavg,z)
    plot_z_ax.plot(task_ic, z)
    plot_z_ax.set_ylim(0,1)
    plot_z_ax.get_yaxis().set_visible(False)
    plot_z_ax.set_xlabel(f"{task_label}")

    fig.savefig(output_path/pathlib.Path(f"{task_name}_hov.png"), dpi=300)
    
args = docopt(__doc__)
df_name = args['<file>']
case = df_name.split('snapshots')[0]
if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = case +'/'
    output_path = pathlib.Path(data_dir).absolute()
tasks = ['b_avg', 'q_avg', 'rh_avg', 'ub_avg', 'uq_avg', 'ux_avg']

with h5py.File(df_name,'r') as df:
    for task in tasks:
        plot_hov(df, task, output_path)
