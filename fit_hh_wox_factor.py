import fit_hh
import jax.numpy as jnp

for i in range(5):
    for factor in 10**jnp.linspace(-1, 1):
        print(factor)
        CONFIG_WOX = fit_hh.MemristorParams(
            # Du 2017 supplement table 1
            alpha   =  1e-8 * 1e6,
            beta    =  0.5,
            gamma   =  1e-5 * 1e6,
            wmin    =  0.1,
            lam     =  1e-3,
            eta     =  8.0,
            delta   =  4.0,
            tau     = 50 * float(factor)
            )
        fit_hh.fit(f'wox_factor_{factor}_try_{i}', CONFIG_WOX, save=True)

