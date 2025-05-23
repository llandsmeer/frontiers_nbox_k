import os

os.environ['CUDA_AVAILABLE_DEVICES'] = ''
#os.environ['CUDA_AVAILABLE_DEVICES'] = '1'
import jax
jax.config.update('jax_default_device', jax.devices('cpu')[0])

import scipy.stats
import jax.numpy as jnp
import tqdm
import numpy as np
import functools
import matplotlib.pyplot as plt
import jax
import cma
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

def exprelr(x): return jnp.where(jnp.isclose(x, 0), 1., x / jnp.expm1(x))
def alpha_m(V): return exprelr(-0.1*V - 4.0)
def alpha_h(V): return 0.07*jnp.exp(-0.05*V - 3.25)
def alpha_n(V): return 0.1*exprelr(-0.1*V - 5.5)
def beta_m(V):  return 4.0*jnp.exp(-(V + 65.0)/18.0)
def beta_h(V):  return 1.0/(jnp.exp(-0.1*V - 3.5) + 1.0)
def beta_n(V):  return 0.125*jnp.exp(-0.0125*V - 0.8125)

TSTOP = 1000

dt = 0.005
SKIP = 1000 * int(0.025/dt)

@functools.partial(jax.jit, static_argnames=['tstop'])
def simhh(tstop=TSTOP):
    v0 = -60
    ou0   =  0.0
    C_m   =   1.0 ;
    E_K   = -77.0 ; E_L   = -53.0 ; E_Na = 50.0
    g_Na  = 120.0 ; g_K   =  36.0 ; g_L  =  0.3
    theta = 0.1   ; sigma = 0.7
    m0 = alpha_m(v0) / (alpha_m(v0) + beta_m(v0))
    h0 = alpha_h(v0) / (alpha_h(v0) + beta_h(v0))
    n0 = alpha_n(v0) / (alpha_n(v0) + beta_n(v0))
    nsteps = int(round(tstop / dt))
    keys = jax.random.split(jax.random.PRNGKey(0), nsteps)
    def f(state, key):
        v, m, h, n, ou = state
        dm_dt = alpha_m(v)*(1-m) - beta_m(v)*m
        dh_dt = alpha_h(v)*(1-h) - beta_h(v)*h
        dn_dt = alpha_n(v)*(1-n) - beta_n(v)*n
        I_K = g_K*n**4*(v - E_K)
        nnext = n + dt * dn_dt
        I_Na = g_Na*m**3*h*(v - E_Na)
        I_L = g_L*(v - E_L)
        Iapp = ou**4
        I_total =  I_Na + I_K + I_L - Iapp
        dv_dt = (1/C_m)*(-I_total)
        return (
            v + dt * dv_dt,
            m + dt * dm_dt,
            h + dt * dh_dt,
            nnext,
            ou + jax.random.normal(key) * sigma * jnp.sqrt(dt) - ou * theta * dt
        ), v
    state0 = v0, m0, h0, n0, ou0
    _, trace = jax.lax.scan(f, state0, keys)
    return trace

@functools.partial(jax.jit, static_argnames=['params'])
def isolve(w, v, params):
    vmem = v
    i = ((1-w) * params.alpha * (1-jnp.exp(-params.beta*vmem)) + w * params.gamma * jnp.sinh(params.delta * vmem))
    return i

@functools.partial(jax.jit, static_argnames=['tstop', 'params', 'return_all'])
def simmem(config, params: MemristorParams, tstop=TSTOP, return_all=False):
    vscale, tscale, iscale = config
    v0 = -60
    ou0   =  0.0
    C_m   =   1.0 ;
    E_K   = -77.0 ; E_L   = -53.0 ; E_Na = 50.0
    g_Na  = 120.0 ; g_L  =  0.3
    theta = 0.1   ; sigma = 0.7
    m0 = alpha_m(v0) / (alpha_m(v0) + beta_m(v0))
    h0 = alpha_h(v0) / (alpha_h(v0) + beta_h(v0))
    w = params.wmin
    #tau = 11.03
    #tau = tau / 5
    # tau = tau * tscale
    nsteps = int(round(tstop / dt))
    keys = jax.random.split(jax.random.PRNGKey(0), nsteps)
    def f(state, key):
        v, m, h, w, ou = state
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
        Iapp = ou**4
        I_total =  I_Na + I_K + I_L - Iapp
        dv_dt = (1/C_m)*(-I_total)
        return (
            v + dt * dv_dt,
            m + dt * dm_dt,
            h + dt * dh_dt,
            wnext,
            ou + jax.random.normal(key) * sigma * jnp.sqrt(dt) - ou * theta * dt
        ), v if not return_all else FullTrace(
                v=v, m=m, h=h, w=w, ou=ou, vmem=vmem, I_K=I_K, I_Na=I_Na, I_L=I_L, Iapp=Iapp, I_total=I_total)
    state0 = v0, m0, h0, w, ou0
    _, trace = jax.lax.scan(f, state0, keys)
    return trace
def main():
    V = simhh()
    v = simmem()
    plt.plot(V, v)
    plt.show()

def fit(name: str, params: MemristorParams, save=True, plot=False, seed=None):
    V = simhh()
    @jax.jit
    def get_scores(vv):
        scores = ((jnp.abs(vv[:,SKIP:] - V[None,SKIP:]))**2).mean(1)
        scores = 0.5 * ((jnp.abs(vv[:,SKIP:] - V[None,SKIP:]))**3).mean(1) + scores
        return scores
    best = cma.optimization_tools.BestSolution()
    run_sims = jax.jit(jax.vmap(jax.jit(functools.partial(simmem, params=params))))
    config ={'bounds': [0.001, 1000], 'popsize': 10 } 
    if seed is not None:
        config['seed'] = seed
    optimizer = cma.CMAEvolutionStrategy(0.2*np.ones(3), .5, config)
    bar = tqdm.tqdm(range(100))
    for _ in bar:
        configs = optimizer.ask()
        vv = run_sims(jnp.array(configs))
        scores = get_scores(vv)
        scores = scores.at[jnp.isnan(scores)].set(0) + jnp.isnan(vv).sum(1) * 1e4
        scores = [float(x) for x in scores]
        optimizer.tell(configs, scores) # type: ignore
        best.update(optimizer.best)
        bar.set_postfix(s=best.f, v=best.x[0], t=best.x[1], i=best.x[2])
    plt.plot(V)
    trace = simmem(best.x, params=params, return_all=True)
    plt.plot(trace.v)
    if save:
        np.savez(f'./out/{name}_hh_fit_results', V=V, config=best.x, score=best.f,
            vscale=best.x[0], tscale=best.x[1], iscale=best.x[2], params=jnp.array(params), **trace._asdict())
    i = 0
    while True:
        fn = f'img/{name}_{i}.png'
        i += 1
        if not os.path.exists(fn):
            break
    if best is not None:
        print(f'''
            score={best.f},
            vscale={best.x[0]}
            tscale={best.x[1]}
            iscale={best.x[2]}
            ''')
    if plot:
        plt.show()
    #else:
        #plt.savefig(fn, dpi=300)
        #plt.savefig(fn.replace('.png', '.svg'))
    ##
    Vbig = simhh(TSTOP*6)
    vbig = simmem(best.x, tstop=TSTOP*6, params=params, return_all=False)
    skip = int(round(TSTOP / dt))
    _, _, r_value, _, _ = scipy.stats.linregress(Vbig[skip:], vbig[skip:])
    r2 = r_value**2
    print(name, r2)

def fit_both():
    fit('wox',  CONFIG_WOX,  plot=0, save=False, seed=592095)
    fit('nbox', CONFIG_NBOX, plot=0, save=False, seed=507062)

if __name__ == '__main__':
    fit_both()
