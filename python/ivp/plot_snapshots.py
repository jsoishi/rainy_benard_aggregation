"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [options]

Options:
    --output=<dir>     Output directory; defaults to 'frames' subdir within the case dir
    --tasks=<tasks>    Tasks to plot [default: b,q,m,rh,b_fluc,q_fluc,m_fluc,rh_fluc,vorticity]
    --3D_default       plots default 3-D fields
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

# Plot settings
dpi = 300

def main(filename, start, count, tasks, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot writes
    with h5py.File(filename, mode='r') as f:
        t = np.array(f['scales/sim_time'])
        for i, task in enumerate(tasks):
            time = t
            center_zero=False
            title = task
            cmap = None
            file_name = task
            if task.startswith('vorticity'):
                center_zero = True
                title = r'$\omega$'
                cmap = 'PuOr'
            elif task.startswith('b'):
                center_zero=False
                cmap = 'RdYlBu_r'
            savename_func = lambda write: '{:s}_{:06d}.png'.format(file_name, write)
            task = f['tasks'][task]
            x = task.dims[1][0][:]
            y = task.dims[2][0][:]
            z = task.dims[3][0][:]
            if x.size == 1:
                x = y
                mask = (0, slice(None), slice(None))
            else:
                mask = (slice(None), 0, slice(None))
            Lz = np.max(z)-np.min(z)
            Lx = np.max(x)-np.min(x)
            height = 1.6
            figsize = (height*(Lx/Lz), height+0.32)
            for k in range(len(t)):
                time = t[k]
                fig, ax = plt.subplots(1, figsize=figsize)
                ax.set_aspect(1)
                pcm = ax.pcolormesh(x, z, task[(k,*mask)].T, shading='nearest',cmap=cmap)
                pmin,pmax = pcm.get_clim()
                if center_zero:
                    # use a CDF to find the
                    H, X = np.histogram(task[(k,*mask)], bins = 100, density = True )
                    dX = X[1] - X[0]
                    F = np.cumsum(H)*dX
                    i_min = np.argmin(np.abs(F-0.05))
                    i_max = np.argmin(np.abs(F-0.95))
                    pmin = X[i_min]
                    pmax = X[i_max]
                    if pmin==0: pmin=-1e-8
                    if pmax==0: pmax=1e-8
                    if pmin > 0: pmin *= -1
                    if pmax < 0: pmax *= -1
                    logger.debug("centering zero: {:.2e} -- 0 -- {:.2e}".format(pmin, pmax))
                    cNorm = matplotlib.colors.TwoSlopeNorm(vmin=pmin, vcenter=0, vmax=pmax)
                else:
                    cNorm = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)
                pcm = ax.pcolormesh(x, z, task[(k,*mask)].T, shading='nearest',cmap=cmap, norm=cNorm)
                ax_cb = fig.add_axes([0.91, 0.4, 0.02, 1-0.4*2])
                cb = fig.colorbar(pcm, cax=ax_cb)
                cb.formatter.set_scientific(True)
                cb.formatter.set_powerlimits((0,4))
                cb.ax.yaxis.set_offset_position('left')
                cb.update_ticks()
                fig.subplots_adjust(left=0.1,right=0.9,top=0.95)
                if title is not None:
                    ax_cb.text(0.5, 1.75, title, horizontalalignment='center', verticalalignment='center', transform=ax_cb.transAxes)
                if time is not None:
                    ax.text(0.975, -0.15, "t = {:.0f}".format(time), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                savename = savename_func(f['scales/write_number'][k])
                savepath = output.joinpath(savename)
                fig.savefig(str(savepath), dpi=dpi)
                #fig.clear()
                plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)
    tasks = args['--tasks'].split(',')
    if args['--3D_default']:
        tasks = ['vorticity y 0.5 y', 'enstrophy 0.5 y', 'm_fluc 0.5 y']
    if args['--output']:
        output_path = pathlib.Path(args['--output']).absolute()
    else:
        case_name = args['<files>'][0].split('snapshots')[0]
        output_path = pathlib.Path(case_name+'/frames').absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, tasks=tasks, output=output_path)
