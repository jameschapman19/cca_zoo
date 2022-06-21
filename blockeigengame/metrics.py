import jax.numpy as jnp
from jax import jit


@jit
def _correct_eigenvector_streak(U, U_correct):
    n_components = U.shape[0]
    cosine_similarities = jnp.diag(
        jnp.corrcoef(U.T, U_correct, rowvar=False)[n_components:, :n_components]
    )
    x_idx = jnp.where(
        jnp.abs(cosine_similarities) > jnp.cos(jnp.pi / 8),
        jnp.ones_like(cosine_similarities),
        jnp.zeros_like(cosine_similarities),
    )
    return jnp.sum(x_idx)


@jit
def _normalized_subspace_distance(U, U_correct):
    U = U.T / jnp.linalg.norm(U, axis=1)
    P = U_correct @ U_correct.T
    U_star = U @ U.T
    return 1 - jnp.trace(U_star @ P) / U_correct.shape[1]


@jit
def _sum_cosine_similarities(U, U_correct):
    n_components = U.shape[0]
    cosine_similarities = jnp.diag(
        jnp.corrcoef(U.T, U_correct, rowvar=False)[n_components:, :n_components]
    )
    return jnp.sum(jnp.abs(cosine_similarities))
