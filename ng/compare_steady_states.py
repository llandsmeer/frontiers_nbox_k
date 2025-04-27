import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import typing

jax.config.update("jax_enable_x64", True)

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

def exprelr(x): return jax.lax.select(jnp.isclose(x, 0), jnp.ones_like(x), x / jnp.expm1(x))
def alpha_m(V): return exprelr(-0.1*V - 4.0)
def alpha_h(V): return 0.07*jnp.exp(-0.05*V - 3.25)
def alpha_n(V): return 0.1*exprelr(-0.1*V - 5.5)
def beta_m(V):  return 4.0*jnp.exp(-(V + 65.0)/18.0)
def beta_h(V):  return 1.0/(jnp.exp(-0.1*V - 3.5) + 1.0)
def beta_n(V):  return 0.125*jnp.exp(-0.0125*V - 0.8125)

dt = 0.005

E_K   = -77.0
g_K   =  36.0

iscale=1.91
tscale=1.26
vscale=0.11

def hh_steady_state(v):
    n = alpha_n(v + E_K) / (alpha_n(v + E_K) + beta_n(v + E_K))
    I_K = g_K*n**4*(v)
    return I_K


def mem_steady_state(v, params: MemristorParams = CONFIG_NBOX):
    w = params.wmin

    vmem = vscale*v

    window_W = 1 - jnp.exp(w*3)/jnp.exp(3)

    wprev = 0
    for _ in range(1000):
        wprev = w
        w = w + window_W * dt * ( params.lam * jnp.sinh(params.eta * vmem) - (w - params.wmin) / params.tau * 5) / tscale
        w = jax.lax.select(w > 0.99, jnp.full_like(w, .99), w)
        w = jax.lax.select(w < params.wmin, jnp.full_like(w, params.wmin), w)
    assert jnp.abs(w - wprev).max() < 1e-3

    I_K = iscale * ((1-w) * params.alpha * (1-jnp.exp(-params.beta*vmem)) + w * params.gamma * jnp.sinh(params.delta * vmem))
    return I_K


V = jnp.linspace(0, 110)
plt.axvline(-60 - -77)
plt.axvline( 20 - -77)
plt.plot(V, hh_steady_state(V)*1e-9*1e6)
plt.plot(V, mem_steady_state(V)*1e-9*1e6)
plt.xlabel('V_K (mV)')
plt.ylabel('I_K (uA/area)')
plt.show()
