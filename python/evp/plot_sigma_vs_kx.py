import sys
from pathlib import Path
import h5py
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

f = mticker.ScalarFormatter(useMathText=True)

plt.style.use('prl')

exclude = None
crit_index = 2
filename = Path(sys.argv[-1])
filedir = filename.parent
fig, ax = plt.subplots(figsize=[8,8/1.6])
with h5py.File(filename, 'r') as df:
    Ra_list = {k:float(k) for k in list(df['curves'].keys())}
    for i,key in enumerate(sorted(Ra_list, key=Ra_list.get, reverse=True)):
        if i == exclude:
            print(f"Skipping Ra ={key}")
        else:
            σ = df[f'curves/{key}']['σ'][:]
            k = df[f'curves/{key}']['k'][:]
            if i == crit_index:
                color = 'C0'
            else:
                color = f'{i/5}'
            ax.semilogx(k, σ.real, label=f'$Ra = {f.format_data(Ra_list[key])}$',color=color)
ax.axhline(0, color='k', linestyle=':',linewidth=1)
ax.legend(fontsize=16)
ax.set_ylim(-0.1,0.1)
ax.set_xlim(0.1,10)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$\omega_r$")
plt.tight_layout()
plot_filename = filedir/Path("growth_curves.pdf")
fig.savefig(plot_filename)
