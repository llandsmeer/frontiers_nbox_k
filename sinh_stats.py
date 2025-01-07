import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('./log.sinh2.db.pkl', 'rb') as f:
    data = pickle.load(f)

out = data['out']

fig, ax = plt.subplots(nrows=4, sharex=True, gridspec_kw=dict(hspace=0))

for k in range(4):
    x = []
    y = []
    for i, block in enumerate(out):
        #if i > 5: continue
        scores = block['scores']
        configs = block['configs']
        xx = configs[:, k]
        #v, t, i, sinh = configs.T
        m = scores < 1000
        x.append(np.log10(xx[m]))
        y.append(scores[m])

    x = np.concatenate(x)
    y = np.concatenate(y)
    #plt.scatter(x, y)
    #ax[k].hexbin(x, -y, gridsize=30, bins='log', mincnt=1)
    ax[k].hist2d(x, -y, bins=30, cmin=1, cmap='winter', alpha=0.8)
    ax[k].set_xlim(-2, 2)
    ax[k].set_ylim(-1000, 0)


plt.savefig('gridsearch_stats.svg')
plt.show()

