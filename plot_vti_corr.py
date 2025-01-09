import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('./log.db.pkl', 'rb') as f:
    data = pickle.load(f)

out = data['out']

transform = lambda x: x

for i, block in enumerate(out):
    scores = block['scores']
    configs = block['configs']
    m = scores < 100
    plt.plot(transform(configs[m,:3]).T, color='blue', zorder=1)
    m2 = ~m & (scores < 150)
    plt.plot(transform(configs[m2,:3]).T, color='green', zorder=-1)

plt.plot(transform(np.array([
    0.14385128,
    0.214236,
    0.194107,
])), color='red', zorder=10)

#plt.ylim(-2, 2)
plt.savefig('./gridsearch_stats_corr.svg')
plt.show()

