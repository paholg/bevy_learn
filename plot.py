#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data.csv')
data.plot(x=0, y=1)
plt.show()
