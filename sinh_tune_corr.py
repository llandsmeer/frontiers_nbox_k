import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('./log.db.pkl', 'rb') as f:
    data = pickle.load(f)

out = data['out']

for i, block in enumerate(out):
    scores = block['scores']
    configs = block['configs']
    m = scores < 100
    plt.plot(np.log10(configs[m]).T, color='red', zorder=1)
    m2 = ~m & (scores < 150)
    plt.plot(np.log10(configs[m2]).T, color='blue', zorder=-1)
    m2 = (~m) & (~m2) & (scores < 200)
    plt.plot(np.log10(configs[m2]).T, color='green', zorder=-1)


plt.ylim(-2, 2)
plt.savefig('./gridsearch_stats_corr.svg')
plt.show()

