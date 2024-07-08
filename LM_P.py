#!/usr/bin/env python
# coding: utf-8

# # Linear Modeling in Python

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

print("BSGP 7030 Linear Modeling in Python")

if len(sys.argv) > 1:
    input_file  = sys.argv[1]
else:
    print("Please provide an input file")
    sys.exit(-1)

df = pd.read_csv(input_file)

print(df.head())

import matplotlib.pyplot as plt

df['x'].head()

df['y'].head()

plt.scatter(df['x'], df['y'])
plt.show()
plt.savefig("PY_ORIG.png")

import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array(df['x']).reshape((-1, 1))
y = np.array(df['y'])

model = LinearRegression()

model.fit(x, y)

intercept = model.intercept_
slope = model.coef_
r_sq = model.score(x,y)

print(f"intercept: {intercept}")
print(f"slope: {slope}")
print(f"r squared: {r_sq}")

y_pred = model.predict(x)

y_pred

plt.plot(df['x'], y_pred)
plt.show()

plt.scatter(df['x'], df['y'])
plt.plot(df['x'], y_pred)
plt.show()

plt.scatter(df['x'], df['y'])
plt.plot(df['x'], y_pred)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
plt.savefig("PY_LM.png")

