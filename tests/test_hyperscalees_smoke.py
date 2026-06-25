import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("hyperscalees") is None,
    reason="HyperscaleES optional dependency is not installed",
)


def test_hyperscalees_eggroll_updates_tiny_mlp():
    import jax
    import jax.numpy as jnp
    import optax
    import hyperscalees as hs
    import operator

    model = hs.models.common.MLP
    noiser = hs.noiser.eggroll.EggRoll
    key = jax.random.key(0)
    model_key, es_key = jax.random.split(key)
    frozen_params, params, scan_map, es_map = model.rand_init(
        model_key, in_dim=3, out_dim=1, hidden_dims=[8], use_bias=True, activation="relu", dtype="float32"
    )
    es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser_params, noiser_params = noiser.init_noiser(params, 0.2, 0.03, solver=optax.sgd, rank=2)

    inputs = jnp.ones((16, 3), dtype=jnp.float32)
    iterinfos = (jnp.zeros((16,), dtype=jnp.int32), jnp.arange(16))
    forward = jax.jit(
        jax.vmap(
            lambda i, x: model.forward(
                noiser, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, i, x
            ),
            in_axes=(0, 0),
        )
    )
    outputs = forward(iterinfos, inputs)
    raw_scores = -((outputs[:, 0] - 2.0) ** 2)
    fitnesses = noiser.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
    noiser_params, new_params = noiser.do_updates(
        frozen_noiser_params, noiser_params, params, es_tree_key, fitnesses, iterinfos, es_map
    )

    diffs = jax.tree.map(lambda a, b: jnp.sqrt(jnp.mean((a - b) ** 2)), params, new_params)
    total_diff = jax.tree.reduce(operator.add, diffs)
    assert total_diff > 0
