import numpy as np
import pandas as pd

Ras = np.array([2e5,1e6,2e6,2e7,5e7,2.5e8])

Vallis_12 = pd.read_csv('beta_1.2.csv', names=['γ', 'Ra_c'])
published_case = np.isclose(Vallis_12['γ'],0.19,atol=1e-2)
Ra_c = Vallis_12['Ra_c'][published_case].values[0]
print(f"Ra_c = {Ra_c:.3g}")
for Ra in Ras:
    print(f'Ra = {Ra:.1g} S = {Ra/Ra_c:.2g}')
