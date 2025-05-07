import jax
import jax.numpy as jnp


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

def f(x):
    return passthrough_clip(x, 0, 1)

print(f(0.5))
print(f(-1))
print(f(2))

print(jax.grad(f)(0.5))
print(jax.grad(f)(-1.))
print(jax.grad(f)(2.))
