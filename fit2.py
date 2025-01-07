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

@jax.jit
def simulate(v, wmin, tau, lam, eta):
    w = wmin
    trace = []
    for _ in range(Npulse):
        # w' = lambda sinh(eta V) - (w - wmin) / tau
        # w' = (tau * lambda sinh(eta V) + wmin - w) / tau
        winf = tau * lam * jnp.sinh(eta * v) + wmin
        w = winf + (w-winf) * jnp.exp(-Tpulse * tau)
        # w = passthrough_clip(w, 0., 1.) # XXX
        trace.append(w)
        w = wmin + (w-wmin) * jnp.exp(-Tinterval * tau)
        trace.append(w)
    trace = jnp.array(trace)
    return trace

def pulseread(v, wmin, tau, lam, eta, alpha, gamma, beta, delta):
    wmin = passthrough_clip(wmin, 0., 1.)
    # wmin = 0.2 # XXX
    w = simulate(v, wmin, tau, lam, eta)
    w = w[::2]
    # return w * 150
    # vread = 0.7
    vread = v
    i = (1-w) * alpha * (1-jnp.exp(-beta*vread)) + w * gamma * jnp.sinh(delta * vread)
    return i, w

@jax.jit
def geti(v, params):
    wmin, tau, lam, eta, alpha, gamma, beta, delta = params
    delta = eta
    i, w = pulseread(v, wmin, tau, lam, eta, alpha, gamma, beta, delta)
    return i, w

@jax.jit
def score(params):
    i, w = jax.vmap(geti, in_axes=[0, None])(vs, params)
    score=((i - itgt)**2).mean()
    score = score + 10 * (params[0]-0.2) ** 2 + 0.1 * (w * (w > 1)).sum() + (w[-1, -1]-.9)**2
    return score

@jax.jit
def update(params, opt_state):
    c = score(params)
    grads = jax.grad(score)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return c, params, opt_state

vs, itgt = load_data()

params = .1 * jnp.ones(8)
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)


for i in tqdm.tqdm(range(1000000)):
    cc, params, opt_state = update(params, opt_state)
    if i % 10000 == 0:
        print(params)
        print(cc)

i, w = jax.vmap(geti, in_axes=[0, None])(vs, params)
#breakpoint()
#plt.show()
plt.plot(i, color='black')
plt.plot(itgt, color='red')
plt.figure()
plt.plot(w, color='black')
#plt.figure()
plt.show()

