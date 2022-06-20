from jax import jit 
import jax.numpy as jnp

@jit
def incrsvd(x_hat,yhat,x_orth,y_orth,U,V,S):
    n_components=U.shape[0]
    Q = jnp.vstack(
        (
            jnp.hstack(
                (
                    jnp.diag(S) + x_hat.T @ yhat,
                    jnp.linalg.norm(y_orth, axis=1).T * x_hat.T,
                )
            ),
            jnp.hstack(
                (
                    (jnp.linalg.norm(x_orth, axis=1).T * yhat.T).T,
                    jnp.atleast_2d(
                        jnp.linalg.norm(x_orth, axis=1, keepdims=True)
                        @ jnp.linalg.norm(y_orth, axis=1, keepdims=True).T
                    ),
                )
            ),
        )
    )
    U_, S, Vt_ = jnp.linalg.svd(Q)
    U = U_[:, :n_components].T @ jnp.vstack((U, x_orth / jnp.linalg.norm(x_orth)))
    V = Vt_.T[:, :n_components].T @ jnp.vstack(
        (V, y_orth / jnp.linalg.norm(y_orth))
    )
    S = S[:n_components]
    return U,V, S