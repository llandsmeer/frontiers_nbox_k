import numpy as np
import matplotlib.pyplot as plt

f = np.load('./out/fit_4g.npz')
i = f['i']
v = f['vs']
itgt = f['itgt']

print(itgt)

#plt.plot(itgt.T, 'o--')
plt.plot(i.T, 'o--')
plt.savefig('out/4gmodel.svg')
plt.ylim(0, 150)
plt.show()
plt.plot(itgt.T, 'o--')
plt.savefig('out/4gdata.svg')
plt.ylim(0, 150)
plt.show()

#print(list(f.keys()))
#input()

