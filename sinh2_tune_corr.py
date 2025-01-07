import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('./log.sinh2.db.pkl', 'rb') as f: 
    data = pickle.load(f)

out = data['out']
vbig = data['Vbig']

vs = []

for i, block in enumerate(out):
    v = block['v']
    vs.append((block['r2'], v))
    scores = block['scores']
    configs = block['configs']
    m = scores < 200
    plt.plot(np.log10(configs[m]).T, color='red', zorder=1)
    m2 = ~m & (scores < 300)
    plt.plot(np.log10(configs[m2]).T, color='blue', zorder=-1)
    m2 = (~m) & (~m2) & (scores < 400)
    plt.plot(np.log10(configs[m2]).T, color='green', zorder=-1)



plt.ylim(-2, 2)
plt.savefig('./gridsearch_stats_corr_sinh2.svg')
plt.show()


vs = sorted(vs)
plt.plot(vbig)
#plt.plot(vs[0][1])
plt.plot(vs[-1][1])
plt.savefig('./gridsearch_stats_sinh2_best_and_worst.svg')
plt.show()
