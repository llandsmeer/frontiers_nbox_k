import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

alpha                 =   0.027066028
beta                  =   0.5026478
gamma                 =  11.137796
wmin                  =   0.11677933
lam                   =   0.015500192
eta                   =   0.7393317
delta                 =   0.7393317

R = 0.01

v = 0.5

def isolve(w, v, Rser):
    iest = 0
    for _ in range(100):
        vr = iest * Rser
        vmem = v - vr
        i = ((1-w) * alpha * (1-jnp.exp(-beta*vmem)) + w * gamma * jnp.sinh(delta * vmem))
        iest = iest * 0.8 + i * 0.2
    return iest


for w in [0.2, 0.5, 0.9]:
    x = jnp.linspace(0, 10)
    i = ((1-w) * alpha * (1-jnp.exp(-beta*x)) + w * gamma * jnp.sinh(delta * x))
    iest = jax.vmap(lambda v: isolve(w, v, R))(x)
    plt.plot(x, i, color='red')
    plt.plot(x, iest, color='blue')
plt.show()

