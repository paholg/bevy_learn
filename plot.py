#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import pandas as pd

if (len(sys.argv) > 1):
    fname = sys.argv[1]
else:
    fname = 'data.csv'

data = pd.read_csv(fname)

fig, ax = plt.subplots()

data.plot(ax=ax, x=0, y=1)
data.plot(ax=ax, x=0, y=2, secondary_y=True)
plt.show()
