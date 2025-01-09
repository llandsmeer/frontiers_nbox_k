import jax.numpy as jnp
import pickle
import tqdm
import scipy.stats
import functools
import matplotlib.pyplot as plt
import jax

alpha                 =   0.027066028
beta                  =   0.5026478
gamma                 =  11.137796
wmin                  =   0.11677933
lam                   =   0.015500192
eta                   =   0.7393317
delta                 =   0.7393317

jax.config.update("jax_enable_x64", True)

def exprelr(x): return jnp.where(jnp.isclose(x, 0), 1., x / jnp.expm1(x))
def alpha_m(V): return exprelr(-0.1*V - 4.0)
def alpha_h(V): return 0.07*jnp.exp(-0.05*V - 3.25)
def alpha_n(V): return 0.1*exprelr(-0.1*V - 5.5)
def beta_m(V):  return 4.0*jnp.exp(-(V + 65.0)/18.0)
def beta_h(V):  return 1.0/(jnp.exp(-0.1*V - 3.5) + 1.0)
def beta_n(V):  return 0.125*jnp.exp(-0.0125*V - 0.8125)

TSTOP = 200

@functools.partial(jax.jit, static_argnames=['tstop'])
def simhh(tstop=TSTOP):
    dt = 0.025
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

@functools.partial(jax.jit, static_argnames=['tstop'])
def simmem(config, tstop=TSTOP):
    # vscale = 2.9e-4
    # tscale = 2.9
    # iscale = .5

    vscale, tscale, iscale, sinscale = config

    dt = 0.025
    v0 = -60
    ou0   =  0.0
    C_m   =   1.0 ;
    E_K   = -77.0 ; E_L   = -53.0 ; E_Na = 50.0
    g_Na  = 120.0 ; g_L  =  0.3
    theta = 0.1   ; sigma = 0.7
    m0 = alpha_m(v0) / (alpha_m(v0) + beta_m(v0))
    h0 = alpha_h(v0) / (alpha_h(v0) + beta_h(v0))
    w = 2.02
    tau = 11.03
    tau = tau * tscale
    wmin = 2.19
    nsteps = int(round(tstop / dt))
    keys = jax.random.split(jax.random.PRNGKey(0), nsteps)
    def f(state, key):
        v, m, h, w, ou = state
        dm_dt = alpha_m(v)*(1-m) - beta_m(v)*m
        dh_dt = alpha_h(v)*(1-h) - beta_h(v)*h

        vmem = vscale*(v - E_K)
        I_K = iscale * ((1-w) * alpha * (1-jnp.exp(-beta*vmem)) + w * gamma * jnp.sinh(delta * vmem))
        I_K = w*(v - E_K)*iscale
        wnext = w + dt * ( lam * jnp.sinh(eta * vmem) - (w - wmin) / tau)

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
        ), v
    state0 = v0, m0, h0, w, ou0
    _, trace = jax.lax.scan(f, state0, keys)
    return trace


def main():
    V = simhh()
    v = simmem()
    plt.plot(V, v)
    plt.show()

def fit():
    V = simhh()
    Vbig = simhh(tstop=1000)
    bestoverall = None
    out = []
    for i in tqdm.tqdm(range(1000)):
        key = jax.random.PRNGKey(i)
        configs = (10**jax.random.uniform(key=key, shape=(5000, 4), minval=-2, maxval=2))
        vv = jax.vmap(simmem)(configs)
        scores = ((jnp.abs(vv - V[None,:]))**2).mean(1)
        m = scores == scores
        scores, configs = scores[m], configs[m]
        idx = scores.argmin()
        best = configs[idx]
        #print(scores[idx])
        #print(best)
        v = simmem(best, tstop=1000)
        vscale, tscale, iscale, sinscale = best
        _, _, r_value, _, _ = scipy.stats.linregress(Vbig[1000:], v[1000:])
        r2 = r_value**2
        if bestoverall is None or bestoverall[0] > scores[idx]:
            bestoverall = scores[idx], r2, f'vscale={vscale} tscale={tscale} iscale={iscale} sinscale={sinscale} R^2={r2}', best, v
        if scores[idx] < 200:
            plt.plot(v, label=f'vscale={vscale} tscale={tscale} iscale={iscale} sinscale={sinscale} R^2={r2}', alpha=0.5)
            print('R^2 =', r2)
        out.append(dict(scores=scores, configs=configs, idx=idx, best=best, r2=r2,
                vscale=best[0], tscale=best[1], iscale=best[2], sinscale=best[3])
                   )
        with open('./4flog.db.pkl', 'wb') as f:
            pickle.dump(dict(
                out=out,
                V=V,
                Vbig=Vbig,
                ), f)
    plt.plot(Vbig, color='black', zorder=-1, lw=3)
    plt.xlim(1000, len(Vbig))
    plt.savefig('fitpo4ft2.svg')
    #plt.show()
    plt.figure()
    plt.plot(Vbig, color='black', zorder=-1, lw=3)
    score, r2, label, best, v = bestoverall
    plt.plot(v)
    plt.title(label)
    plt.savefig('fitpot4f3.svg')
    #plt.hist(scores, bins=100)
    #plt.show()

if __name__ == '__main__':
    fit()
