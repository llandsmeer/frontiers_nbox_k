import matplotlib.pyplot as plt
import jax.numpy as jnp
import fit_du2015_to_ju2024

t, g = fit_du2015_to_ju2024.load_data()
plt.plot(t, g, 'o')

period = 1.1

ax = plt.gca().twinx()

for i in range(49):
    ax.hlines(.5, i*period, i*period+1)
    ax.hlines(0, i*period+1, i*period+period)
ax.hlines(0, 51*period, 100*period)

plt.savefig('./out/ju2024.svg')
plt.show()
