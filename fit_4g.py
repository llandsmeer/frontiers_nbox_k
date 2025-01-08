import pandas as pd
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
    tangent_out = x_dot # + a_dot + b_dot
    return primal_out, tangent_out

def load_data():
    df = pd.read_csv('./data/JuKim24_G.csv', header=[0,1])
    vnames = sorted(list(set(df.columns.get_level_values(0))))
    Itgt = []
    vs = []
    for i, vname in enumerate(vnames):
        xy = df[vname].sort_values('X')
        x = xy.X.values
        y = xy.Y.values
        assert len(x) == Npulse
        assert all(x.round() == np.arange(1, len(x)+1))
        Itgt.append(y)
        vs.append(float(vname.removeprefix('V-')))
        if i > 3:
            break
    vs = jnp.array(vs)
    itgt = jnp.array(Itgt)
    return vs, itgt

@jax.jit
def simulate(v, wmin, tau, lam, eta):
    w = wmin
    trace = []
    for _ in range(Npulse):
        # w' = lambda sinh(eta V) - (w - wmin) / tau
        # w' = (tau * lambda sinh(eta V) + wmin - w) / tau
        winf = tau * lam * jnp.sinh(eta * v) + wmin
        w = winf + (w-winf) * jnp.exp(-Tpulse / tau)
        # w = passthrough_clip(w, 0., 1.) # XXX
        trace.append(w)
        w = wmin + (w-wmin) * jnp.exp(-Tinterval / tau)
        trace.append(w)
    trace = jnp.array(trace)
    return trace

def w_to_i(w, vread, alpha, beta, gamma, delta):
    return (1-w) * alpha * (1-jnp.exp(-beta*vread)) + w * gamma * jnp.sinh(delta * vread)


def pulseread(v, wmin, tau, lam, eta, alpha, gamma, beta, delta):
    wmin = passthrough_clip(wmin, 0., 1.)
    # wmin = 0.2 # XXX
    w = simulate(v, wmin, tau, lam, eta)
    w = w[::2]
    # return w * 150
    # vread = 0.7
    vread = v
    # w = passthrough_clip(w, 0., 1.)
    i = w_to_i(w, vread, alpha, beta, gamma, delta)
    return i, w

@jax.jit
def geti(v, params):
    wmin, lam, eta, alpha, gamma, beta = params
    delta = eta
    i, w = pulseread(v, wmin, tau, lam, eta, alpha, gamma, beta, delta)
    return i, w

@jax.jit
def score(params):
    wmin, _, eta, alpha, gamma, beta = params
    delta = eta
    i, w = jax.vmap(geti, in_axes=[0, None])(vs, params)
    score=((i - itgt)**2).mean()
    score = score + 1000 * (wmin-0.15) ** 2 + 100 * (w * (w > 1)).sum()# + 0.01*(w[-1, -1]-.8)**2
    # 4a constraints
    G0_0_7V = 2.18# 2.044595956802368
    imodel07 = w_to_i(wmin, 0.7, alpha, beta, gamma, delta)
    idata07 = G0_0_7V * 0.7
    score = score + 1*(imodel07 - idata07) ** 2
    score = score + 10000 * (jnp.abs(params) * (params < 0)).sum()
    score = score + 1*(beta - 0.5) ** 2
    #jax.debug.print('{} {}', imodel07, idata07)
    return score

@jax.jit
def update(params, opt_state):
    c = score(params)
    grads = jax.grad(score)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return c, params, opt_state

vs, itgt = load_data()

params = .1 * jnp.ones(6)
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)


for i in tqdm.tqdm(range(1000000)):
    cc, params, opt_state = update(params, opt_state)
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
plt.figure()
plt.plot(w.T, color='black')
np.savez('./out/fit_4g3',
        i=i,
        itgt=itgt,
        w=w,
        vs=vs,
        ** {name:params[i] for i, name in enumerate('wmin lam eta alpha gamma beta'.split())},
        params=params
)
#plt.figure()
plt.show()

# alpha                     0.027066028
# beta                      0.5026478
# gamma                    11.137796
# wmin                      0.11677933
# lam                       0.015500192
# eta                       0.7393317

# alpha         1e-8        0.01
# beta          0.5         0.5
# gamma         1e-5        10
# delta         4.0         4.0
# lambda        1e-3        1e-3
# eta           8.          8.
# tau           0.05        0.05
