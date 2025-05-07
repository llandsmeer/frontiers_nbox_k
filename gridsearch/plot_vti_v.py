import pickle
import numpy as np
import matplotlib.pyplot as plt

name = 'log.db.pkl'

with open(name, 'rb') as f:
    data = pickle.load(f)

best = None
out = data['out']

transform = lambda x: x

plt.plot(data['Vbig'], color='black')
for i, block in enumerate(out):
    scores = block['scores']
    configs = block['configs']
    v = block['v']
    idxbest = scores.argmin()
    if best is None or scores[idxbest] < best[0]:
        best = scores[idxbest], v, configs[idxbest]
    if block['scores'].min() < 250:
        plt.plot(v)

plt.savefig(f'./vti_v_{name}.svg')
plt.clf()
plt.plot(best[1])
plt.plot(data['Vbig'], color='black')
plt.savefig(f'./vti_v_BEST_{name}.svg')
np.savez(f'./traces_v_vti_BEST_{name}', tgt=data['Vbig'], v=best[1], s=best[0], cfg=best[2])
#plt.show()


