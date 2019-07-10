import numpy as np
import matplotlib.pyplot as plt


from numpy import loadtxt 

reward =loadtxt("epsilons3.dat", comments="#", delimiter=",", unpack=False)

lens=len(reward)
episodes = [i for i in range(lens)]

fig = plt.figure(figsize=(9,8))


# markers_on = [i for i in range(0,lens,100)]
# x = range(len(episodes))
x=[i for i in range(0,lens,100)]

# plt.xticks(x,  episodes)
plt.xticks(x,  x)
locs, labels = plt.xticks()
plt.setp(labels, rotation=70)
# plt.plot(x, reward, '-gD', markevery=markers_on)
plt.plot(episodes, reward, '-bo')

reward_mean=reward.reshape(13, 100)[:,-1]
x2=[i+100 for i in range(0,lens,100)]
print(x)
print(x2)
plt.plot(x2, reward_mean, '-rD')

fig.suptitle('epsilon change', fontsize=20)
plt.xlabel('episodes', fontsize=15)
plt.ylabel('epsilon', fontsize=15)

# plt.plot([0, 1300], [20000, 20000], 'k--', lw=3)

# plt.axis(episodes)


plt.xticks(rotation=70)

plt.savefig('2.png')
plt.show()