import pandas as pd
import functools
import tqdm
import optax
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

Tpulse    = 1 # 1ms
Tinterval = 1 # 1ms
Npulse    = 10

tau = 11.73

@jax.custom_jvp
def passthrough_clip(x: float | jax.Array, a, b):
    return jnp.clip(x, a, b)

@passthrough_clip.defjvp
def passthrough_clip_jvp(primals, tangents):
    x, a, b = primals
    x_dot, a_dot, b_dot = tangents
    del a_dot
    del b_dot
    primal_out = jnp.clip(x, a, b)
    tangent_out = x_dot
    return primal_out, tangent_out

def load_data():
    df = pd.read_csv('./data/JuKim24_G.csv', header=[0,1])
    vnames = sorted(list(set(df.columns.get_level_values(0))))
    Itgt = []
    vs = []
    for vname in vnames:
        xy = df[vname].sort_values('X')
        x = xy.X.values
        y = xy.Y.values
        assert len(x) == Npulse
        assert all(x.round() == np.arange(1, len(x)+1))
        Itgt.append(y)
        vs.append(float(vname.removeprefix('V-')))
    vs = jnp.array(vs)
    itgt = jnp.array(Itgt)
    return vs, itgt

@functools.partial(jax.jit, static_argnames=['clip'])
def simulate(v, wmin, tau, lam, eta, clip=False):
    w = wmin
    trace = []
    for _ in range(Npulse):
        # w' = lambda sinh(eta V) - (w - wmin) / tau
        # w' = (tau * lambda sinh(eta V) + wmin - w) / tau
        window_W = 1 - jnp.exp(w*3)/jnp.exp(3)
        tau_bar = tau / window_W
        winf = tau_bar * lam * jnp.sinh(eta * v) + wmin
        w = winf + (w-winf) * jnp.exp(-Tpulse / tau_bar)
        if clip:
            w = passthrough_clip(w, 0., 1.) # XXX
        trace.append(w)
        w = wmin + (w-wmin) * jnp.exp(-Tinterval / tau)
        trace.append(w)
    trace = jnp.array(trace)
    return trace

def w_to_i(w, vread, alpha, beta, gamma, delta):
    return (1-w) * alpha * (1-jnp.exp(-beta*vread)) + w * gamma * jnp.sinh(delta * vread)


def pulseread(v, wmin, tau, lam, eta, alpha, gamma, beta, delta, clip=False):
    wmin = passthrough_clip(wmin, 0., 1.)
    # wmin = 0.2 # XXX
    w = simulate(v, wmin, tau, lam, eta, clip=clip)
    w = w[::2]
    # return w * 150
    # vread = 0.7
    vread = v
    # w = passthrough_clip(w, 0., 1.)
    i = w_to_i(w, vread, alpha, beta, gamma, delta)
    return i, w

@functools.partial(jax.jit, static_argnames=['clip'])
def geti(v, params, clip=False):
    wmin, lam, eta, alpha, gamma, beta = params
    delta = eta
    i, w = pulseread(v, wmin, tau, lam, eta, alpha, gamma, beta, delta, clip=clip)
    return i, w

@functools.partial(jax.jit, static_argnames=['clip'])
def score(params, clip=False):
    wmin, _, eta, alpha, gamma, beta = params
    delta = eta
    i, w = jax.vmap(lambda v, p: geti(v, p, clip=clip), in_axes=[0, None])(vs, params)
    score =  ((i - itgt)**2).mean()
    score = score + 500 * (params[0]-0.1) ** 2
    score = score + 100 * (w * (w > 1)).sum()
    # 4a constraints
    G0_0_7V = 2.044595956802368
    imodel07 = w_to_i(wmin, 0.7, alpha, beta, gamma, delta)
    idata07 = G0_0_7V * 0.7
    score = score + 1*(imodel07 - idata07) ** 2
    score = score + 10000 * (jnp.abs(params) * (params < 0)).sum()
    score = score + 0.1*(beta - 0.5) ** 2
    #jax.debug.print('{} {}', imodel07, idata07)
    return score

@functools.partial(jax.jit, static_argnames=['clip'])
def update(params, opt_state, clip=False):
    c = score(params)
    grads = jax.grad(lambda p: score(p, clip=clip))(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return c, params, opt_state

vs, itgt = load_data()

params = .1 * jnp.ones(6)
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)


for i in tqdm.tqdm(range(1000000)):
    cc, params, opt_state = update(params, opt_state, clip=False)
    if i % 10000 == 0:
        print(params)
        print(cc)

for i, name in enumerate('wmin lam eta alpha gamma beta'.split()):
    print(name.ljust(10), params[i])

i, w = jax.vmap(geti, in_axes=[0, None])(vs, params)
#breakpoint()
#plt.show()
plt.plot(i.T, color='black')
plt.plot(itgt.T, color='red')
plt.savefig('out/fit4g.svg')
plt.figure()
plt.plot(w.T, color='black')
#plt.figure()
