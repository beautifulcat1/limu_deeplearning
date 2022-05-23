#学习matplot

from cProfile import label
import matplotlib.pyplot as plt
fmts=('-', 'm--', 'g-.', 'r:')
x = [1,2,3,4,5]
y = [2,3,4,5,6]
y1 = [3,4,5,6,7]
y2 = [4,5,6,7,5]
plt.plot(x,y,'-')
plt.plot(x,y1,'m--')
plt.plot(x,y2,'g-.')
plt.show()