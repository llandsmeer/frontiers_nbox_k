import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('./log.db.pkl', 'rb') as f:
    data = pickle.load(f)

out = data['out']

transform = lambda x: x

for i, block in enumerate(out):
    scores = block['scores']
    idx = scores.argmin()
    print(list(block.keys()))


    print(list(block.keys()))

input()
plt.show()

