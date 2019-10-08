"""
Plot profiles from joint analysis files.

Usage:
    plot_profiles.py <filename> 

"""

import h5py
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from docopt import docopt
from dedalus.extras import plot_tools
plt.style.use('prl')

# parse arguments
args = docopt(__doc__)
filename = Path(args['<filename>'])

outbase = Path("plots")

#title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
#savename_func = lambda write: 'mid_{:06}.png'.format(write)
# Layout
nrows, ncols = 2, 3
scale = 2.5

image = plot_tools.Box(2, 2)
pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

# Create multifigure
mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
fig = mfig.figure

tasks= ['b', 'q', 'temp', 'u', 'v', 'w']
even_scale = [False, False, False, True, True, True]
with h5py.File(filename, 'r') as df:
    for n, task in enumerate(tasks):
        i, j = divmod(n, ncols)
        axes = mfig.add_axes(i, j, [0, 0, 1, 1])
        dset = df['tasks'][task]
        image_axes = (0,3)
        data_slices = (slice(None), 0, 0, slice(None))
        plot_tools.plot_bot(dset, image_axes, data_slices, axes=axes, title=task, even_scale=even_scale[n])
    
for f in filename.parts:
    if f.startswith('rainy_benard'):
        plot_base = f
plot_file_name = Path(plot_base + '_profiles.png')

plt.savefig(outbase/plot_file_name, dpi=300)
