import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import json
import jax.numpy as jnp
import jax.scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

period = 1.1

def load_data():
    pot = pd.read_csv('data/A_Pot.csv', header=None, names=['pulsenum', 'conductance_muS'])
    pot.conductance_muS = 7 - pot.conductance_muS
    tpot = period * (pot.pulsenum-1)
    gpot = pot.conductance_muS
    dec = pd.read_csv('data/A_Dec.csv', header=None, names=['pulsenum', 'conductance_muS'])
    dec.conductance_muS = 7 - dec.conductance_muS
    tdec = period * (dec.pulsenum-1)
    gdec = dec.conductance_muS
    gdec = gdec[tdec.argsort()]
    tdec = tdec[tdec.argsort()]
    t = jnp.concatenate([tpot.values, tdec.values])
    g = jnp.concatenate([gpot.values, gdec.values])
    return t, g

def load_data_repeat(n):
    T, G = [], []
    t, g = load_data()
    for i in range(n):
        T.append(t + i*period*100)
        G.append(g)
    return jnp.concatenate(T), jnp.concatenate(G)

@jax.jit
def simulate(w0, A, tau, wmin, dt):
    def wnext(w, A):
        return w + dt * (A - (w-wmin)/tau), w
    _, trace = jax.lax.scan(wnext, w0, A)
    return trace

def model(w0, tau, amp, wmin):
    dt = 0.01
    t  = jnp.arange(0, int(round(NREPEAT*period*100)), dt)
    #$A = (t%(100*period) < 55) & (t % period < 1.)
    A = (t%(100*period) < 54) & (t % period < 1.)
    trace = simulate(w0, amp*A, tau, wmin, dt)
    return t[::10], trace[::10]

def cost(params):
    w0, tau, amp, wmin = params
    t, g = model(w0, tau, amp, wmin)
    Tg = jnp.interp(T, t, g)
    return jnp.mean((Tg - G)**2)

NREPEAT = 10
T, G = load_data_repeat(NREPEAT)

def fit_model_to_data():
    initial_guess = jnp.array([2.0, 20.0, 2.0, 2.5])
    initial_guess = jnp.array([2.52, 18.8, 0.22, 1.73])
    initial_guess = jnp.array([1.0, 1.0, 1.0, 1.])

    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_guess)

    @jax.jit
    def update(params, opt_state):
        c = cost(params)
        grads = jax.grad(cost)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return c, params, opt_state

    num_steps = 10000
    params = initial_guess
    pbar = tqdm(range(num_steps), desc="Optimizing", unit="step")
    ls = []
    for step in pbar:
        current_cost, params, opt_state = update(params, opt_state)
        ls.append(current_cost)
        if step % 100 == 0:
            pbar.set_postfix(cost=current_cost)

    plt.plot(ls)
    plt.xlabel('Iteration')
    plt.savefig('out/fit_du2015_to_ju2024_loss.svg')
    plt.yscale('log')
    plt.savefig('out/fit_du2015_to_ju2024_loss_logscale.svg')
    plt.show()

    print("Optimized parameters:", params)

    w0, tau, amp, wmin = params

    with open('out/fit_du2015_to_ju2024.json', 'w') as f:
        json.dump(dict(
            w0=float(w0),
            tau=float(tau),
            amp=float(amp),
            wmin=float(wmin),
            ), f)

    t, g = model(w0, tau, amp, wmin)

    plt.plot(T, G, label="Observed", color="black", lw=2)
    plt.plot(t, g, label="Optimized", color="red", ls="--")
    plt.xlabel("T")
    plt.ylabel("G")
    plt.legend()
    plt.title("Optimized vs Observed")
    plt.savefig('out/fit_du2015_to_ju2024.svg')
    plt.show()

if __name__ == '__main__':
    fit_model_to_data()
