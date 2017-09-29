import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('TargetPositions.txt', delimiter=r"\s+", header=None)
data.columns = ["Time", "x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "x4", "y4", "z4", "xo", "yo", "zo"]

x_cord = data["xo"].as_matrix()
y_cord = data["yo"].as_matrix()
plt.plot(x_cord,y_cord)
x_cord = data["x1"].as_matrix()
y_cord = data["y1"].as_matrix()
plt.plot(x_cord,y_cord)
x_cord = data["x2"].as_matrix()
y_cord = data["y2"].as_matrix()
plt.plot(x_cord,y_cord)
x_cord = data["x3"].as_matrix()
y_cord = data["y3"].as_matrix()
plt.plot(x_cord,y_cord)
x_cord = data["x4"].as_matrix()
y_cord = data["y4"].as_matrix()
plt.plot(x_cord,y_cord)
ax = plt.gca()
ax.grid(True)
plt.show()
