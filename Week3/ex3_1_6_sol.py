#%%
import os
from xcorr import xcorr
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# load positive cases data
positives_file = os.path.join(DATA_DIR, 'Test_pos_over_time.csv')
positives_data = np.genfromtxt(positives_file, delimiter=';', dtype=str)
i_start = np.where(positives_data[:, 0] == '01/10/2020')[0][0]
i_end = np.where(positives_data[:, 0] == '28/02/2021')[0][0]
positives_data = positives_data[i_start:i_end, 1].astype(int)

# load hospitalize cases data
hospitalized_file = os.path.join(DATA_DIR, 'Newly_admitted_over_time.csv')
hospitalized_data = np.genfromtxt(hospitalized_file, delimiter=';', dtype=str)
i_start = np.where(hospitalized_data[:, 0] == '2020-10-01')[0][0]
i_end = np.where(hospitalized_data[:, 0] == '2021-02-28')[0][0]
hospitalized_data = hospitalized_data[i_start:i_end, -1].astype(int)

# calculate and plot cross-correlation
k = 25
xk = np.arange(-k,k+1)
ccf = xcorr(hospitalized_data, positives_data, k)

fig, ax = plt.subplots(3, 1, figsize=(14, 8))
ax[0].plot(positives_data)
ax[0].set_title('Positive cases')
ax[0].set_xlabel('Time (days)')
ax[1].plot(hospitalized_data)
ax[1].set_title('Hospitalized cases')
ax[1].set_xlabel('Time (days)')
ax[2].bar(xk, ccf)
ax[2].set_title('Cross-correlation')
ax[2].set_xlabel('Lag (days)')

plt.tight_layout()
plt.show()