import os

#os.environ['CUDA_AVAILABLE_DEVICES'] = ''
#import jax
#jax.config.update('jax_default_device', jax.devices('cpu')[0])

import jax.numpy as jnp
import functools
import matplotlib.pyplot as plt
import jax
import typing

class MemristorParams(typing.NamedTuple):
    alpha   : float
    beta    : float
    gamma   : float
    wmin    : float
    lam     : float
    eta     : float
    delta   : float
    tau     : float

CONFIG_NBOX = MemristorParams(
    # results from fit_4g.py
    alpha   =  1.0378567,
    beta    =  1.2025751,
    gamma   =  13.483879,
    wmin    =  0.082101814,
    lam     =  0.015609243,
    eta     =  0.71460414,
    delta   =  0.71460414,
    tau     =  11.03
)

CONFIG_WOX = MemristorParams(
    # Du 2017 supplement table 1
    alpha   =  1e-8 * 1e6,
    beta    =  0.5,
    gamma   =  1e-5 * 1e6,
    wmin    =  0.1,
    lam     =  1e-3,
    eta     =  8.0,
    delta   =  4.0,
    tau     = 50
    )

class FullTrace(typing.NamedTuple):
    v: float
    m: float
    h: float
    w: float
    ou: float
    vmem: float
    I_K: float
    I_Na: float
    I_L: float
    Iapp: float
    I_total: float

jax.config.update("jax_enable_x64", True)

def exprelr(x): return jax.lax.select(jnp.isclose(x, 0), jnp.ones_like(x), x / jnp.expm1(x))
def alpha_m(V): return exprelr(-0.1*V - 4.0)
def alpha_h(V): return 0.07*jnp.exp(-0.05*V - 3.25)
def alpha_n(V): return 0.1*exprelr(-0.1*V - 5.5)
def beta_m(V):  return 4.0*jnp.exp(-(V + 65.0)/18.0)
def beta_h(V):  return 1.0/(jnp.exp(-0.1*V - 3.5) + 1.0)
def beta_n(V):  return 0.125*jnp.exp(-0.0125*V - 0.8125)

TSTOP = 1000

dt = 0.005
SKIP = 1000 * int(0.025/dt)

@functools.partial(jax.jit, static_argnames=['params'])
def isolve(w, v, params):
    vmem = v
    i = ((1-w) * params.alpha * (1-jnp.exp(-params.beta*vmem)) + w * params.gamma * jnp.sinh(params.delta * vmem))
    return i

v0 = -60
ou0   =  0.0
theta = 0.1   ; sigma = 0.7
theta = 0.1   ; sigma = 0.2
C_m   =   1.0 ;
E_K   = -77.0 ; E_L   = -53.0 ; E_Na = 50.0
g_Na  = 120.0 ; g_L  =  0.3

def single_neuron_timestep(v, m, h, w, Iapp):
    dm_dt = alpha_m(v)*(1-m) - beta_m(v)*m
    dh_dt = alpha_h(v)*(1-h) - beta_h(v)*h
    vmem = vscale*(v - E_K)
    I_K = iscale * isolve(w, vmem, params)
    window_W = 1 - jnp.exp(w*3)/jnp.exp(3)
    wnext = w + window_W * dt * ( params.lam * jnp.sinh(params.eta * vmem) - (w - params.wmin) / params.tau * 5) / tscale
    wnext = jax.lax.select(wnext > 0.99, .99, wnext)
    wnext = jax.lax.select(wnext < params.wmin, params.wmin, wnext)
    I_Na = g_Na*m**3*h*(v - E_Na)
    I_L = g_L*(v - E_L)
    I_total =  I_Na + I_K + I_L - Iapp
    dv_dt = (1/C_m)*(-I_total)
    return (
        v + dt * dv_dt,
        m + dt * dm_dt,
        h + dt * dh_dt,
        wnext,
    )

@functools.partial(jax.jit, static_argnames=['tstop', 'params', 'ncomparts'])
def simmem(params: MemristorParams, tstop=TSTOP, ncomparts=1):
    m0 = alpha_m(v0) / (alpha_m(v0) + beta_m(v0))
    h0 = alpha_h(v0) / (alpha_h(v0) + beta_h(v0))
    w = params.wmin
    nsteps = int(round(tstop / dt))
    Gcompart = 0.1
    keys = jax.random.split(jax.random.PRNGKey(0), nsteps)
    def f(state, key):
        v, m, h, w, ou = state
        Iapp = jnp.zeros_like(v)
        Iapp = Iapp.at[0].add(ou**4)
        Iapp = Iapp.at[:-1].add(
                (v[1:] - v[:-1]) * Gcompart
                )
        Iapp = Iapp.at[1:].add(
                (v[:-1] - v[1:]) * Gcompart
                )
        v, m, h, w = jax.vmap(single_neuron_timestep)(v, m, h, w, Iapp)
        return (v, m, h, w,
            ou + jax.random.normal(key) * sigma * jnp.sqrt(dt) - ou * theta * dt
        ), v
    state0 = v0, m0, h0, w, ou0
    init = tuple(jnp.full((ncomparts,), x) if i < 4 else x for i, x in enumerate(state0))
    _, trace = jax.lax.scan(f, init, keys)
    return trace

if 1:
    params = CONFIG_NBOX
    vscale=0.11
    tscale=1.26
    iscale=1.91
else:
    params = CONFIG_WOX
    tscale=0.186
    vscale=0.013
    iscale=6.317

best = jnp.array([vscale, tscale, iscale])

cmap = plt.get_cmap('RdBu')
trace = simmem(params=params, ncomparts=30)
for i in range(30):
    plt.plot(jnp.arange(len(trace)) * dt, -i + trace[:,i], color=cmap(0.1+0.9*(i/29)),label=f'comp{i}')
plt.xlim(70, 250)
plt.legend()
# plt.savefig('./out/circuit.svg')
plt.show()

# (5_w,10)-aCMA-ES (mu_w=3.2,w_1=45%) in dimension 3 (seed=592095, Mon May  5 14:01:17 2025)
# 100%|██████████████████| 100/100 [01:04<00:00,  1.56it/s, i=6.32, s=3.75e+3, t=0.186, v=0.0131]
#             score=3752.687685030218,
#             vscale=0.013061788087031106
#             tscale=0.18595913291576718
#             iscale=6.316548044931718
