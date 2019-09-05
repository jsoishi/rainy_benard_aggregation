"""plot_energy.py<filename>

Usage:
    plot_energy.py <filename>

"""

import h5py
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from docopt import docopt

plt.style.use('prl')

# parse arguments
args = docopt(__doc__)
filename = Path(args['<filename>'])

outbase = Path("plots")
df = h5py.File(filename, 'r')

plt.plot(df['scales/sim_time'], df['tasks/KE'][:,0,0,0])
plt.xlabel("time")
plt.ylabel("Kinetic Energy")

plot_file_name = Path(filename.stem + '_ke_timeseries.png')
plt.tight_layout()
plt.savefig(outbase/plot_file_name, dpi=300)
