import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('./log.db.pkl', 'rb') as f:
    data = pickle.load(f)

out = data['out']

transform = lambda x: x

best = None
for i, block in enumerate(out):
    scores = block['scores']
    configs = block['configs']
    idxbest = scores.argmin()
    if best is None or scores[idxbest] < best[0]:
        best = scores[idxbest], configs[idxbest]
    m = scores < 150
    plt.plot(transform(configs[m]).T, color='blue', zorder=1)
    m2 = ~m & (scores < 200)
    plt.plot(transform(configs[m2]).T, color='green', zorder=-1)

if best is not None:
    plt.plot(transform(best[1]), color='red', zorder=10)

#plt.ylim(-2, 2)
plt.savefig('./gridsearch_stats_corr.svg')
#plt.show()

