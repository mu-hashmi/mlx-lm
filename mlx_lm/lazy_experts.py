import json
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .models.switch_layers import QuantizedSwitchLinear


class LazyQuantizedSwitchLinear(nn.Module):
    """Drop-in replacement for QuantizedSwitchLinear that loads experts on demand.

    On each forward call: loads a fresh lazy ref from the safetensors shard,
    slices only the needed experts, and lets the full tensor be freed after
    evaluation. This avoids the problem where evaluating a slice from a lazy
    tensor permanently materializes the full source tensor in Metal memory.
    """

    def __init__(self, shard_path: str, key_prefix: str, group_size: int,
                 bits: int, mode: str):
        super().__init__()
        self._shard_path = shard_path
        self._key_prefix = key_prefix
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.freeze()

    def _load_expert_subset(self, expert_ids: mx.array):
        """Load only the needed experts from the safetensors shard."""
        shard = mx.load(self._shard_path)
        w = shard[f"{self._key_prefix}.weight"][expert_ids]
        s = shard[f"{self._key_prefix}.scales"][expert_ids]
        biases_key = f"{self._key_prefix}.biases"
        b = shard[biases_key][expert_ids] if biases_key in shard else None
        return w, s, b

    def __call__(self, x, indices, sorted_indices=False):
        mx.eval(indices)
        indices_np = np.asarray(indices.reshape(-1))
        unique_ids = np.unique(indices_np)

        w, s, b = self._load_expert_subset(mx.array(unique_ids))

        remap = np.empty(int(unique_ids[-1]) + 1, dtype=np.int32)
        remap[unique_ids] = np.arange(len(unique_ids), dtype=np.int32)
        remapped = mx.array(remap[indices_np].reshape(indices.shape))

        return mx.gather_qmm(
            x,
            w,
            s,
            b,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


def _build_shard_map(model_path: Path) -> dict[str, str]:
    """Read model.safetensors.index.json and return {key: absolute_shard_path}."""
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    return {key: str(model_path / shard) for key, shard in weight_map.items()}


def enable_lazy_experts(model, model_path: Path) -> int:
    """Replace QuantizedSwitchLinear modules in MoE layers with lazy versions.

    Args:
        model: The loaded MLX model (with lazy=True).
        model_path: Path to the model directory containing safetensors shards.

    Returns:
        Number of modules replaced (expected: 48 layers x 3 = 144).
    """
    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)

    replaced = 0
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        switch = layer.mlp.switch_mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(switch, name)
            if not isinstance(orig, QuantizedSwitchLinear):
                continue
            key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            lazy = LazyQuantizedSwitchLinear(
                shard_path=shard_path,
                key_prefix=key_prefix,
                group_size=orig.group_size,
                bits=orig.bits,
                mode=orig.mode,
            )
            setattr(switch, name, lazy)
            replaced += 1
    return replaced
