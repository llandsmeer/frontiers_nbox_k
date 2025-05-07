import numpy as np
import jax.numpy as jnp
import scipy.signal
import matplotlib.pyplot as plt
import fit_hh
plt.rcParams['svg.fonttype'] = 'none'

nbox = np.load('./out/nbox_hh_fit_results.npz')
wox  = np.load('./out/wox_hh_fit_results.npz')

dt = 0.025
l = len(nbox['V'])

N = l // 4*3
t = np.arange(l-N) * dt

print(N)
plt.plot(t, nbox['V'][N:])
plt.plot(t, nbox['v'][N:], label='nbox')
plt.plot(t, wox['v'][N:], label='wox')

plt.plot([0, 100], [30, 30], color='black', lw=3)
plt.plot([100, 100], [-80, 40], color='black', lw=3)

for i in [-80, -40, 0, 40]:
    plt.plot([90, 100], [i, i], color='black', lw=3)
    plt.text(60, i, f'{i} mV', va='center')
    plt.text(100, i, f'NbOx {i * nbox["vscale"]:.1f}V', va='center')
    plt.text(140, i, f'WOx {i * wox["vscale"]:.1f}V', va='center')

E_K   = -77.0 ; E_L   = -53.0 ; E_Na = 50.0
plt.axhline(E_K)
plt.axhline(E_Na)
plt.axhline(E_L)

plt.text(0, 0, f'NbOx {100 * nbox["tscale"]:.1f}ms\nWOx {100 * wox["tscale"]:.1f}ms')
plt.ylim(-80, 60)

plt.legend()
plt.axis('off')
plt.savefig('./out/hh_both.svg')
plt.xlim(167, 196)
plt.savefig('./out/hh_zoom.svg')
#plt.show()


def power(trace):
    I_SI = np.abs(trace['I_K'] * 1e-9)
    V_SI = np.abs(trace['vmem'] * 1e-3)
    INa_SI = 1/trace['iscale'] * np.abs(trace['I_Na'] * 1e-9)
    VNa_SI = trace['vscale'] * np.abs(trace['v'] - E_Na) * 1e-3
    IL_SI = 1/trace['iscale'] * np.abs(trace['I_L'] * 1e-9)
    VL_SI = trace['vscale'] * np.abs(trace['v'] - E_L) * 1e-3
    return I_SI * V_SI, I_SI * V_SI + INa_SI * VNa_SI + IL_SI * VL_SI

plt.clf()

ee = []
ee2 = []
text = ''
for name, trace, prom in [('nbox', nbox, 1e-9), ('wox', wox, 1e-10)]:
    pwr, pwrfull = power(trace)
    Ekall = np.trapz(pwr, dx=dt *trace['tscale'])
    #plt.plot(pwr)
    Efull = np.trapz(pwrfull, dx=dt *trace['tscale'])
    pwr = pwr[N:]
    pwrfull = pwrfull[N:]
    text = text + f'{name} K {Ekall*1e9:.1f} nW\n'
    text = text + f'{name} All {Efull*1e6:.3f} uW\n'
    peaks, _ = scipy.signal.find_peaks(-pwr, distance=100, prominence=prom)
    peaks = np.concatenate([[0], peaks, [-1]])
    es = []
    es2 = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        #plt.plot(pwr[a:b])
        E = np.trapz(pwr[a:b], dx=dt *trace['tscale'])
        es.append(E)
        E = np.trapz(pwrfull[a:b], dx=dt *trace['tscale'])
        es2.append(E)
    ee.append(es)
    print(name)
    print(np.mean(es), np.std(es), len(es))
    ee2.append(es2)
    print(np.mean(es2), np.std(es2), len(es2))


#plt.show()

plt.boxplot(ee)
plt.xticks([1,2], ['nbox', 'wox'])
#plt.ylim(-.1e-9, 2.1e-9)
plt.text(1.5, 1e-9, text)
plt.plot([.8, .8], [0, 2e-9], color='black', lw=3)
plt.plot([.7, .8], [2e-9, 2e-9], color='black', lw=3)
plt.plot([.7, .8], [1e-9, 1e-9], color='black', lw=3)
plt.plot([.7, .8], [0e-9, 0e-9], color='black', lw=3)
#plt.axis('off')

plt.savefig('out/kperspike.svg')
plt.clf()

plt.boxplot(ee2)
plt.xticks([1,2], ['nbox', 'wox'])
#plt.ylim(1.9e-9, 6.1e-9)
plt.text(1.5, 1e-9, text)
plt.plot([.8, .8], [2e-9, 6e-9], color='black', lw=3)
plt.plot([.7, .8], [6e-9, 6e-9], color='black', lw=3)
plt.plot([.7, .8], [4e-9, 4e-9], color='black', lw=3)
#plt.plot([.7, .8], [1e-8, 1e-8], color='black', lw=3)
plt.plot([.7, .8], [2e-9, 2e-9], color='black', lw=3)

plt.savefig('out/allperspike.svg')
plt.clf()


# vmem = 10

# window_W = 1 - jnp.exp(w*3)/jnp.exp(3)
# params = fit_hh.CONFIG_NBOX
# wnext = w + window_W * dt * ( params.lam * jnp.sinh(params.eta * vmem) - (params.w - params.wmin) / params.tau) / tscale
#window_W = 1 - jnp.exp(w*3)/jnp.exp(3)
#tau_bar = tau / window_W
#winf = tau_bar * lam * jnp.sinh(eta * v) + wmin

#plt.show()
