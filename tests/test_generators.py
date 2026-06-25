import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None or importlib.util.find_spec("hyperscalees") is None,
    reason="JAX and HyperscaleES optional dependencies are required",
)


def test_direct_generators_emit_layout_shaped_logits():
    import hyperscalees as hs
    import jax
    import jax.numpy as jnp

    from pixelant_eggroll.config import GeneratorConfig, LayoutSpec
    from pixelant_eggroll.models_jax import generator_class

    noiser = hs.noiser.eggroll.EggRoll
    for kind in ("mlp", "cnn"):
        layout = LayoutSpec(height=16, width=16, feed_pixels=((7, 0), (8, 0)))
        config = GeneratorConfig(kind=kind, layout=layout, hidden_dims=(16, 8), latent_dim=4, spectrum_dim=81)
        init = generator_class(kind).rand_init(jax.random.key(0), config)
        es_tree_key = hs.models.common.simple_es_tree_key(init.params, jax.random.key(1), init.scan_map)
        frozen_noiser_params, noiser_params = noiser.init_noiser(init.params, 0.1, 0.01, rank=1)

        logits = generator_class(kind).forward(
            noiser,
            frozen_noiser_params,
            noiser_params,
            init.frozen_params,
            init.params,
            es_tree_key,
            (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
            jnp.ones((81,), dtype=jnp.float32),
            jnp.ones((4,), dtype=jnp.float32),
        )

        assert logits.shape == (1, 16, 16)


def test_mlp_generator_accepts_deeper_hidden_dims():
    import hyperscalees as hs
    import jax
    import jax.numpy as jnp

    from pixelant_eggroll.config import GeneratorConfig, LayoutSpec
    from pixelant_eggroll.models_jax import DirectMLPGenerator

    noiser = hs.noiser.eggroll.EggRoll
    layout = LayoutSpec(height=8, width=8, feed_pixels=((3, 0), (4, 0)))
    config = GeneratorConfig(kind="mlp", layout=layout, hidden_dims=(16, 12, 8), spectrum_dim=81)
    init = DirectMLPGenerator.rand_init(jax.random.key(0), config)
    es_tree_key = hs.models.common.simple_es_tree_key(init.params, jax.random.key(1), init.scan_map)
    frozen_noiser_params, noiser_params = noiser.init_noiser(init.params, 0.1, 0.01, rank=1)

    logits = DirectMLPGenerator.forward(
        noiser,
        frozen_noiser_params,
        noiser_params,
        init.frozen_params,
        init.params,
        es_tree_key,
        (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
        jnp.ones((81,), dtype=jnp.float32),
    )

    assert logits.shape == (1, 8, 8)


def test_pngf_cnn_generator_noiseless_eval_projects_to_binary_center_feed():
    import jax
    import jax.numpy as jnp

    from pixelant_eggroll.config import GeneratorConfig
    from pixelant_eggroll.models_jax import DirectCNNGenerator, direct_cnn_eval
    from pixelant_eggroll.pngf import PNGF_LAYOUT, PNGF_TARGET_DIM, hard_project_pngf_center_fed

    config = GeneratorConfig(kind="cnn", layout=PNGF_LAYOUT, hidden_dims=(8,), latent_dim=0, spectrum_dim=PNGF_TARGET_DIM)
    init = DirectCNNGenerator.rand_init(jax.random.key(0), config, channels=8)
    logits = direct_cnn_eval(init.params, init.frozen_params, jnp.ones((3, PNGF_TARGET_DIM), dtype=jnp.float32))
    masks = hard_project_pngf_center_fed(logits)

    assert logits.shape == (3, 1, 21, 21)
    assert masks.shape == (3, 1, 21, 21)
    assert masks[0, 0, 10, 9] == 1.0
    assert masks[0, 0, 10, 11] == 1.0
    assert masks[0, 0, 10, 10] == 0.0


def test_generator_checkpoint_round_trips_for_evaluation(tmp_path):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from pixelant_eggroll.checkpoints import load_inverse_npz, save_inverse_npz
    from pixelant_eggroll.config import GeneratorConfig
    from pixelant_eggroll.models_jax import DirectCNNGenerator, direct_cnn_eval
    from pixelant_eggroll.pngf import PNGF_LAYOUT, PNGF_TARGET_DIM

    config = GeneratorConfig(kind="cnn", layout=PNGF_LAYOUT, hidden_dims=(4,), latent_dim=0, spectrum_dim=PNGF_TARGET_DIM)
    init = DirectCNNGenerator.rand_init(jax.random.key(0), config, channels=4)
    path = tmp_path / "gen.npz"
    save_inverse_npz(path, init.frozen_params, init.params, {"stage": "test"})
    frozen, params, metadata = load_inverse_npz(path)

    x = jnp.ones((2, PNGF_TARGET_DIM), dtype=jnp.float32)
    original = direct_cnn_eval(init.params, init.frozen_params, x)
    loaded = direct_cnn_eval(params, frozen, x)

    assert metadata["stage"] == "test"
    assert frozen["layout"] == (1, 21, 21)
    assert np.allclose(np.asarray(original), np.asarray(loaded))
