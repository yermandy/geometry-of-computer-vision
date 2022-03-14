from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt



ax: Axes = plt.axes(projection='3d')
C = np.array([0.1, 0.1, 0.1])
Base = np.eye(3)
Cx, Cy, Cz = Base.T

ax.quiver(*C, *Cx, arrow_length_ratio=0.1, color='r')
ax.quiver(*C, *Cy, arrow_length_ratio=0.1, color='g')
ax.quiver(*C, *Cz, arrow_length_ratio=0.1, color='b')

ax.text(*(Cx + C), f'x')
ax.text(*(Cy + C), f'y')
ax.text(*(Cz + C), f'z')

ax.scatter(0, 0, 0, color='black')

plt.tight_layout()
plt.show()
