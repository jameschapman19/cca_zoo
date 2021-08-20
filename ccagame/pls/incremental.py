# Importing necessary libraries

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit

from .utils import TV, initialize


# Update rule to be used for calculating eigenvectors
#@partial(jit, static_argnums=(2))
def update(X, Y, U, S,V, lr=0.1):
    uhat=X.T@U
    u_orth=X-U.T@U@X
    vhat=Y.T@V
    v_orth=Y-V.T@V@Y
    Q=np.vstack(np.hstack(np.diag(S)+uhat.T@vhat, jnp.linalg.norm(v_orth)*uhat.T),np.hstack(jnp.linalg.norm(u_orth)*vhat, jnp.linalg.norm(u_orth)*jnp.linalg.norm(v_orth)))
    U_,S,V_=jnp.linalg.svd(Q)
    U=jnp.hstack(U,u_orth/jnp.linalg.norm(u_orth))@U_.T
    V = jnp.hstack(V, v_orth / jnp.linalg.norm(v_orth)) @ V_.T
    return U,S,V

# Run the update step iteratively across all eigenvectors
def calc_incremental(X, Y, k, iterations=100,
              random_state=0):
    U, V = initialize(X, Y, k, 'random', random_state)
    S=np.zeros(k)
    for i in range(iterations):
        U,S,V = update(X,Y,U,S,V)
        print(f'iteration {i}: {TV(X, Y, U, V)}')
    return TV(X, Y, U, V), U, V