import jax.numpy as jnp
import scipy.signal
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

TSTOP = 2000

dt = 0.005
SKIP = 1000 * int(0.025/dt)

@functools.partial(jax.jit, static_argnames=['params'])
def isolve(w, v, params):
    vmem = v
    i = ((1-w) * params.alpha * (1-jnp.exp(-params.beta*vmem)) + w * params.gamma * jnp.sinh(params.delta * vmem))
    return i

v0 = -60
C_m   =   1.0 ;
E_K   = -77.0 ; E_L   = -53.0 ; E_Na = 50.0
g_Na  = 120.0 ; g_K   =  36.0 ; g_L  =  0.3

def simhh(iapp, tstop=TSTOP):
    m0 = alpha_m(v0) / (alpha_m(v0) + beta_m(v0))
    h0 = alpha_h(v0) / (alpha_h(v0) + beta_h(v0))
    n0 = alpha_n(v0) / (alpha_n(v0) + beta_n(v0))
    nsteps = int(round(tstop / dt))
    keys = jax.random.split(jax.random.PRNGKey(0), nsteps)
    def f(state, _):
        v, m, h, n = state
        dm_dt = alpha_m(v)*(1-m) - beta_m(v)*m
        dh_dt = alpha_h(v)*(1-h) - beta_h(v)*h
        dn_dt = alpha_n(v)*(1-n) - beta_n(v)*n
        I_K = g_K*n**4*(v - E_K)
        nnext = n + dt * dn_dt
        I_Na = g_Na*m**3*h*(v - E_Na)
        I_L = g_L*(v - E_L)
        I_total =  I_Na + I_K + I_L - iapp
        dv_dt = (1/C_m)*(-I_total)
        return (
            v + dt * dv_dt,
            m + dt * dm_dt,
            h + dt * dh_dt,
            nnext
        ), v
    state0 = v0, m0, h0, n0
    _, trace = jax.lax.scan(f, state0, keys)
    return trace

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

@functools.partial(jax.jit, static_argnames=['tstop', 'params'])
def simmem(params: MemristorParams, tstop=TSTOP, *, Iapp_const):
    m0 = alpha_m(v0) / (alpha_m(v0) + beta_m(v0))
    h0 = alpha_h(v0) / (alpha_h(v0) + beta_h(v0))
    w = params.wmin
    nsteps = int(round(tstop / dt))
    keys = jax.random.split(jax.random.PRNGKey(0), nsteps)
    def f(state, _):
        v, m, h, w = jax.vmap(single_neuron_timestep)(*state, Iapp=Iapp_const)
        return (v, m, h, w), v
    state0 = v0, m0, h0, w
    init = tuple(jnp.full((Iapp_const.shape[0],), x) for x in state0)
    _, trace = jax.lax.scan(f, init, keys)
    return trace

def calc_f(arr):
    crossings = (arr[:-1] < -40) & (arr[1:] > -40)
    return crossings.sum()

params = CONFIG_NBOX

iscale=1.91
tscale=1.26
vscale=0.11

best = jnp.array([vscale, tscale, iscale])

i = jnp.linspace(0, 20)
trace = simmem(params=CONFIG_NBOX, Iapp_const=i)
trace = trace[trace.shape[0]//2:]

trace_hh = jax.vmap(simhh)(i).T
trace_hh = trace_hh[trace_hh.shape[0]//2:]

f_mem = jax.vmap(calc_f)(trace.T)
f_hh = jax.vmap(calc_f)(trace_hh.T)
plt.plot(i, f_mem, 'o--', label='nbox')
plt.plot(i, f_hh, 'o--', label='hh')
plt.legend()
plt.savefig('./out/fI.svg')
plt.show()
