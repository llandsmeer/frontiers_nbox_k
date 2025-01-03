import jax
import functools
import pandas as pd
import scipy.optimize
import json
import matplotlib.pyplot as plt
import jax.numpy as jnp
import fit_du2015_to_ju2024

with open('./out/fit_du2015_to_ju2024.json', 'r') as f:
    data = json.load(f)
    params = jnp.array([
        data['w0'],
        data['tau'],
        data['amp'],
        data['wmin']
    ])

@functools.partial(jax.jit, static_argnames=['pos', 'idx'])
def find_bounds(delta, params, pos, idx, bound):
    delta = jnp.abs(delta)
    if not pos:
        delta = delta * -1
    delta = 1 + delta
    params = params.at[idx].set(params[idx] * delta)
    return cost(params) - bound

cost = jax.jit(fit_du2015_to_ju2024.cost)

mcost = cost(params)

Nvary = 1000
change = 3

bound = mcost * 1.1 # 10% deviation

names = 'w0', 'tau', 'amp', 'wmin'

df = []
idx = 0
for idx in range(4):
    bpos = scipy.optimize.root_scalar(find_bounds, args=(params, True, idx, bound), bracket=[0, .7], x0=0.1)
    bneg = scipy.optimize.root_scalar(find_bounds, args=(params, False, idx, bound), bracket=[0, .7], x0=0.1)
    mult = jnp.ones((Nvary, 4))
    mult = mult.at[:, idx].set(2**jnp.linspace(-jnp.log2(change), +jnp.log2(change), Nvary))
    pvar = mult * params[None,:]
    costs = jax.vmap(cost)(pvar)
    #plt.plot(mult, costs)
    print(bpos.converged)

    #plt.axvline(1 + bpos.root, color='black')
    #plt.axhline(mcost*1.1)
    #plt.axvline(1 - bneg.root, color='black')
    rneg = params[idx]*(1-bneg.root)
    rpos = params[idx]*(1+bpos.root)
    df.append(dict(
        #idx=idx,
        name=names[idx],
        val=params[idx],
        #eneg=params[idx]*bneg.root,
        #epos=params[idx]*bpos.root,
        range_neg=rneg,
        range_pos=rpos
        ))
    if idx != 0:
        plt.hlines(idx, -bneg.root*100, +bpos.root*100)
        plt.text(0, idx, f'{names[idx]} {params[idx]:.2f} {rneg:.2f} {rpos:.2f}')

df = pd.DataFrame(df)
df.to_csv('./out/fit_du2015_to_ju2024_sensitivity.csv')
df.to_latex('./out/fit_du2015_to_ju2024_sensitivity.tex')

print(df)
#plt.axhline(mcost*1.1)
#plt.xlim(0, 2)
plt.axvline(1)

plt.savefig('./out/fit_du2015_to_ju2024_sensitivity.svg')
plt.show()
