import numpy as np
import matplotlib.pyplot as plt


from numpy import loadtxt 

reward =loadtxt("test_rewards3.dat", comments="#", delimiter=",", unpack=False)

lens=len(reward)
episodes = [i for i in range(lens)]

fig = plt.figure(figsize=(9,8))


# markers_on = [i for i in range(0,lens,100)]
x = range(len(episodes))
x=[i for i in range(0,lens,10)]

plt.xticks(x,  episodes)
plt.xticks(x,  x)
locs, labels = plt.xticks()
plt.setp(labels, rotation=70)
# plt.plot(x, reward, '-gD', markevery=markers_on)
plt.plot(episodes, reward, '-bo')


fig.suptitle('grades change', fontsize=20)
plt.xlabel('episodes', fontsize=15)
plt.ylabel('grades', fontsize=15)

plt.plot([0, 100], [20000, 20000], 'k--', lw=3)

# plt.axis(episodes)


plt.xticks(rotation=70)

plt.savefig('3.png')
plt.show()