#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import pandas as pd

if (len(sys.argv) > 1):
    fname = sys.argv[1]
else:
    fname = 'data.csv'

data = pd.read_csv(fname)
data['cumulative reward'] = data['reward'].cumsum()

fig, ax = plt.subplots()

data.plot(ax=ax, x='episode', y='cumulative reward')
data.plot(ax=ax, x='episode', y='steps', secondary_y=True)
plt.show()
