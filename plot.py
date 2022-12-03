#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data.csv')

fig, ax = plt.subplots()

data.plot(ax=ax, x=0, y=1)
data.plot(ax=ax, x=0, y=2, secondary_y=True)
plt.show()
