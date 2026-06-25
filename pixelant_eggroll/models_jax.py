"""JAX model definitions matching the antenna notebooks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from .config import DEFAULT_LAYOUT, GeneratorConfig, LayoutSpec


def _hs_common():
    try:
        from hyperscalees.models.base_model import CommonInit, CommonParams, Model
        from hyperscalees.models.common import EXCLUDED, MM_PARAM, PARAM, call_submodule, merge_frozen, merge_inits
        from hyperscalees.models.common import Linear as HSLinear
    except ImportError as exc:
        raise ImportError(
            "HyperscaleES is required for EGGROLL training. Install optional dependencies with "
            "`pip install -r requirements-eggroll.txt`."
        ) from exc
    return CommonInit, CommonParams, Model, HSLinear, call_submodule, merge_inits, merge_frozen, PARAM, MM_PARAM, EXCLUDED


def leaky_relu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.leaky_relu(x, negative_slope=0.01)


class BatchNorm1d:
    @classmethod
    def rand_init(cls, dim: int, dtype: str = "float32"):
        CommonInit, *_rest = _hs_common()
        PARAM = _rest[-3]
        params = {
            "gamma": jnp.ones((dim,), dtype=dtype),
            "beta": jnp.zeros((dim,), dtype=dtype),
        }
        frozen_params = {
            "mean": jnp.zeros((dim,), dtype=dtype),
            "var": jnp.ones((dim,), dtype=dtype),
            "eps": jnp.array(1e-5, dtype=dtype),
        }
        return CommonInit(frozen_params, params, {"gamma": (), "beta": ()}, {"gamma": PARAM, "beta": PARAM})

    @classmethod
    def _forward(cls, common_params, x: jnp.ndarray) -> jnp.ndarray:
        gamma = common_params.noiser.get_noisy_standard(
            common_params.frozen_noiser_params,
            common_params.noiser_params,
            common_params.params["gamma"],
            common_params.es_tree_key["gamma"],
            common_params.iterinfo,
        )
        beta = common_params.noiser.get_noisy_standard(
            common_params.frozen_noiser_params,
            common_params.noiser_params,
            common_params.params["beta"],
            common_params.es_tree_key["beta"],
            common_params.iterinfo,
        )
        mean = common_params.frozen_params["mean"]
        var = common_params.frozen_params["var"]
        eps = common_params.frozen_params["eps"]
        return (x - mean) * jax.lax.rsqrt(var + eps) * gamma + beta


class InverseGenerator:
    """Spectrum-to-logits generator used by the EGGROLL inverse phase."""

    @classmethod
    def rand_init(
        cls,
        key,
        hidden_dims: Sequence[int] = (1054, 512),
        dtype: str = "float32",
        input_dim: int = 81,
        output_dim: int = 144,
    ):
        CommonInit, _CommonParams, _Model, HSLinear, _call, merge_inits, merge_frozen, *_ = _hs_common()
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")
        keys = jax.random.split(key, len(hidden_dims) + 1)
        in_dim = input_dim
        layers = {}
        for idx, hidden_dim in enumerate(hidden_dims):
            layers[f"fc{idx}"] = HSLinear.rand_init(keys[idx], in_dim, hidden_dim, True, dtype)
            layers[f"bn{idx}"] = BatchNorm1d.rand_init(hidden_dim, dtype)
            in_dim = hidden_dim
        layers["out"] = HSLinear.rand_init(keys[-1], in_dim, output_dim, True, dtype)
        merged = merge_inits(**layers)
        return merge_frozen(merged, hidden_dims=tuple(hidden_dims), input_dim=input_dim, output_dim=output_dim)

    @classmethod
    def forward(
        cls,
        noiser,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        iterinfo,
        spectrum: jnp.ndarray,
    ) -> jnp.ndarray:
        _CommonInit, CommonParams, _Model, HSLinear, call_submodule, *_ = _hs_common()
        common = CommonParams(noiser, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, iterinfo)
        x = spectrum
        for idx in range(len(frozen_params["hidden_dims"])):
            x = call_submodule(HSLinear, f"fc{idx}", common, x)
            x = leaky_relu(_call_batchnorm(common, f"bn{idx}", x))
        return call_submodule(HSLinear, "out", common, x)


class DirectMLPGenerator:
    """Layout-aware MLP generator with the legacy inverse-network architecture."""

    @classmethod
    def rand_init(
        cls,
        key,
        config: GeneratorConfig,
        dtype: str = "float32",
    ):
        init = InverseGenerator.rand_init(
            key,
            hidden_dims=config.hidden_dims,
            dtype=dtype,
            input_dim=config.spectrum_dim + config.latent_dim,
            output_dim=config.layout.flat_size,
        )
        frozen = dict(init.frozen_params)
        frozen["layout"] = config.layout.mask_shape
        frozen["latent_dim"] = config.latent_dim
        frozen["spectrum_dim"] = config.spectrum_dim
        return type(init)(frozen, init.params, init.scan_map, init.es_map)

    @classmethod
    def forward(
        cls,
        noiser,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        iterinfo,
        spectrum: jnp.ndarray,
        latent: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        x = _condition_input(spectrum, latent)
        logits = InverseGenerator.forward(
            noiser,
            frozen_noiser_params,
            noiser_params,
            frozen_params,
            params,
            es_tree_key,
            iterinfo,
            x,
        )
        return logits.reshape(frozen_params["layout"])


class DirectCNNGenerator:
    """Spectrum/noise-conditioned spatial generator for larger binary layouts."""

    @classmethod
    def rand_init(
        cls,
        key,
        config: GeneratorConfig,
        dtype: str = "float32",
        channels: int | None = None,
    ):
        CommonInit, _CommonParams, _Model, HSLinear, _call, merge_inits, merge_frozen, PARAM, *_ = _hs_common()
        keys = jax.random.split(key, 5)
        layout = config.layout
        channels = channels or int(config.hidden_dims[0])
        input_dim = config.spectrum_dim + config.latent_dim
        projection_dim = channels * layout.height * layout.width
        merged = merge_inits(
            proj=HSLinear.rand_init(keys[0], input_dim, projection_dim, True, dtype),
            conv0=_conv_init(keys[1], channels, channels, dtype),
            conv1=_conv_init(keys[2], channels, channels, dtype),
            out=_conv_init(keys[3], channels, layout.layers, dtype),
        )
        frozen = dict(merged.frozen_params)
        frozen["layout"] = layout.mask_shape
        frozen["latent_dim"] = config.latent_dim
        frozen["spectrum_dim"] = config.spectrum_dim
        frozen["channels"] = channels
        frozen["height"] = layout.height
        frozen["width"] = layout.width
        return CommonInit(frozen, merged.params, merged.scan_map, merged.es_map)

    @classmethod
    def forward(
        cls,
        noiser,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        iterinfo,
        spectrum: jnp.ndarray,
        latent: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        _CommonInit, CommonParams, _Model, HSLinear, call_submodule, *_ = _hs_common()
        common = CommonParams(noiser, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, iterinfo)
        x = _condition_input(spectrum, latent)
        x = call_submodule(HSLinear, "proj", common, x)
        x = x.reshape((frozen_params["channels"], frozen_params["height"], frozen_params["width"]))
        x = leaky_relu(_conv_noisy(common, "conv0", x))
        x = leaky_relu(_conv_noisy(common, "conv1", x))
        return _conv_noisy(common, "out", x)


def generator_class(kind: str):
    if kind == "mlp":
        return DirectMLPGenerator
    if kind == "cnn":
        return DirectCNNGenerator
    raise ValueError(f"Unknown generator kind {kind!r}; expected 'mlp' or 'cnn'")


def _condition_input(spectrum: jnp.ndarray, latent: jnp.ndarray | None) -> jnp.ndarray:
    if latent is None or latent.shape[-1] == 0:
        return spectrum
    return jnp.concatenate([spectrum, latent], axis=-1)


def _conv_init(key, in_channels: int, out_channels: int, dtype: str = "float32", kernel_size: int = 3):
    CommonInit, *_rest = _hs_common()
    PARAM = _rest[-3]
    fan_in = in_channels * kernel_size * kernel_size
    scale = jnp.sqrt(jnp.asarray(2.0 / fan_in, dtype=dtype))
    weight = jax.random.normal(key, (out_channels, in_channels, kernel_size, kernel_size), dtype=dtype) * scale
    bias = jnp.zeros((out_channels,), dtype=dtype)
    params = {"weight": weight, "bias": bias}
    return CommonInit({}, params, {"weight": (), "bias": ()}, {"weight": PARAM, "bias": PARAM})


def _conv_noisy(common_params, name: str, x: jnp.ndarray) -> jnp.ndarray:
    _CommonInit, CommonParams, *_ = _hs_common()
    sub_common = CommonParams(
        common_params.noiser,
        common_params.frozen_noiser_params,
        common_params.noiser_params,
        common_params.frozen_params[name],
        common_params.params[name],
        common_params.es_tree_key[name],
        common_params.iterinfo,
    )
    weight = sub_common.noiser.get_noisy_standard(
        sub_common.frozen_noiser_params,
        sub_common.noiser_params,
        sub_common.params["weight"],
        sub_common.es_tree_key["weight"],
        sub_common.iterinfo,
    )
    bias = sub_common.noiser.get_noisy_standard(
        sub_common.frozen_noiser_params,
        sub_common.noiser_params,
        sub_common.params["bias"],
        sub_common.es_tree_key["bias"],
        sub_common.iterinfo,
    )
    y = jax.lax.conv_general_dilated(
        x[None, :, :, :],
        weight,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y[0] + bias[:, None, None]


def _call_batchnorm(common_params, name: str, x: jnp.ndarray) -> jnp.ndarray:
    _CommonInit, CommonParams, *_ = _hs_common()
    sub_common = CommonParams(
        common_params.noiser,
        common_params.frozen_noiser_params,
        common_params.noiser_params,
        common_params.frozen_params[name],
        common_params.params[name],
        common_params.es_tree_key[name],
        common_params.iterinfo,
    )
    return BatchNorm1d._forward(sub_common, x)


def bn_eval(x: jnp.ndarray, bn: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    return (x - bn["mean"]) * jax.lax.rsqrt(bn["var"] + bn.get("eps", 1e-5)) * bn["gamma"] + bn["beta"]


def linear_eval(x: jnp.ndarray, layer: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    return x @ layer["weight"].T + layer["bias"]


def direct_mlp_eval(
    params: Mapping[str, Any],
    frozen_params: Mapping[str, Any],
    spectrum: jnp.ndarray,
    latent: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Noiseless batched evaluation for `DirectMLPGenerator` parameters."""

    x = _condition_input(jnp.asarray(spectrum, dtype=jnp.float32), latent)
    if x.ndim == 1:
        x = x[None, :]
    for idx in range(len(frozen_params["hidden_dims"])):
        x = leaky_relu(bn_eval(linear_eval(x, params[f"fc{idx}"]), frozen_params[f"bn{idx}"]))
    logits = linear_eval(x, params["out"])
    return logits.reshape((x.shape[0],) + tuple(int(v) for v in frozen_params["layout"]))


def direct_cnn_eval(
    params: Mapping[str, Any],
    frozen_params: Mapping[str, Any],
    spectrum: jnp.ndarray,
    latent: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Noiseless batched evaluation for `DirectCNNGenerator` parameters."""

    x = _condition_input(jnp.asarray(spectrum, dtype=jnp.float32), latent)
    if x.ndim == 1:
        x = x[None, :]
    channels = int(frozen_params["channels"])
    height = int(frozen_params["height"])
    width = int(frozen_params["width"])
    x = linear_eval(x, params["proj"]).reshape((x.shape[0], channels, height, width))
    x = leaky_relu(conv2d_same_nchw(x, params["conv0"]))
    x = leaky_relu(conv2d_same_nchw(x, params["conv1"]))
    return conv2d_same_nchw(x, params["out"])


def conv2d_same_nchw(x: jnp.ndarray, layer: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    y = jax.lax.conv_general_dilated(
        x,
        layer["weight"],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y + layer["bias"][None, :, None, None]


def batchnorm2d_eval(x: jnp.ndarray, bn: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    return (x - bn["mean"][None, :, None, None]) * jax.lax.rsqrt(
        bn["var"][None, :, None, None] + bn.get("eps", 1e-5)
    ) * bn["gamma"][None, :, None, None] + bn["beta"][None, :, None, None]


def forward_surrogate(params: Mapping[str, Any], design: jnp.ndarray) -> jnp.ndarray:
    """Forward CNN surrogate matching `Net_big` in the notebooks."""

    x = design
    for idx in range(1, 17):
        x = conv2d_same_nchw(x, params[f"conv{idx}"])
        x = leaky_relu(batchnorm2d_eval(x, params[f"bn{idx}"]))
    x = x.reshape((x.shape[0], -1))
    x = leaky_relu(bn_eval(linear_eval(x, params["fc17"]), params["bn17"]))
    x = leaky_relu(bn_eval(linear_eval(x, params["fc18"]), params["bn18"]))
    return linear_eval(x, params["fc19"])
