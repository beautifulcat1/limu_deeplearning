import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 10)
y1 = 2 * t
y2 = -4 * t + 5

plt.subplot(211)
plt.plot(t, y1, label='ax1')
plt.legend()
plt.subplot(212)
plt.plot(t, y2, label='ax2')
plt.legend()
plt.show()