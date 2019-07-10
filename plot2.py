import numpy as np
import matplotlib.pyplot as plt


from numpy import loadtxt 

action= loadtxt("actions2.dat", comments="#", delimiter=",", unpack=False)

fig, ax = plt.subplots()  

hist, bins = np.histogram(action, bins=4)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)



fig.suptitle('episodes numbers for each action', fontsize=20)

plt.xlabel('action', fontsize=15)
# plt.ylabel('grades', fontsize=15)

xx=[u'no action', u'right arm', u'middle', u'left arm']

xxi=[0.35, 1.1, 1.85, 2.65]

for i, v in enumerate(xx):
    plt.text(xxi[i]-0.2, 180, str(v), color='black')

plt.show()
