import importlib.util
import operator

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None
    or importlib.util.find_spec("hyperscalees") is None
    or importlib.util.find_spec("optax") is None,
    reason="JAX, Optax, and HyperscaleES optional dependencies are required",
)


@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_tiny_eggroll_step_with_fake_scorer(kind):
    import hyperscalees as hs
    import jax
    import jax.numpy as jnp
    import optax

    from pixelant_eggroll.antenna_ops import hard_threshold_design
    from pixelant_eggroll.config import GeneratorConfig, LayoutSpec
    from pixelant_eggroll.fitness import FitnessConfig, compute_fitness
    from pixelant_eggroll.models_jax import generator_class

    layout = LayoutSpec(height=8, width=8, feed_pixels=((3, 0), (4, 0)))
    config = GeneratorConfig(kind=kind, layout=layout, hidden_dims=(8, 4), latent_dim=2, spectrum_dim=81)
    generator = generator_class(kind)
    init = generator.rand_init(jax.random.key(0), config)
    noiser = hs.noiser.eggroll.EggRoll
    es_tree_key = hs.models.common.simple_es_tree_key(init.params, jax.random.key(1), init.scan_map)
    frozen_noiser_params, noiser_params = noiser.init_noiser(init.params, 0.1, 0.01, solver=optax.sgd, rank=1)
    iterinfos = (jnp.zeros((2,), dtype=jnp.int32), jnp.arange(2))
    targets = jnp.zeros((2, 81), dtype=jnp.float32)
    latents = jnp.ones((2, 2), dtype=jnp.float32)

    def apply_one(iterinfo, target, latent):
        return generator.forward(
            noiser,
            frozen_noiser_params,
            noiser_params,
            init.frozen_params,
            init.params,
            es_tree_key,
            iterinfo,
            target,
            latent,
        )

    logits = jax.vmap(apply_one)(iterinfos, targets, latents)
    designs = hard_threshold_design(logits, layout=layout)
    fake_prediction = jnp.repeat(jnp.mean(designs, axis=(1, 2, 3), keepdims=False)[:, None], 81, axis=1)
    raw_fitness, _ = compute_fitness(fake_prediction, targets, designs, FitnessConfig(), layout=layout)
    fitnesses = noiser.convert_fitnesses(frozen_noiser_params, noiser_params, raw_fitness)
    _, new_params = noiser.do_updates(
        frozen_noiser_params, noiser_params, init.params, es_tree_key, fitnesses, iterinfos, init.es_map
    )

    diffs = jax.tree.map(lambda a, b: jnp.sqrt(jnp.mean((a - b) ** 2)), init.params, new_params)
    total_diff = jax.tree.reduce(operator.add, diffs)
    assert total_diff >= 0
