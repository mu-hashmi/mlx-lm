import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .models.switch_layers import QuantizedSwitchLinear


# ---------------------------------------------------------------------------
# Model-agnostic helpers
# ---------------------------------------------------------------------------

def _find_switch_mlp(layer, layer_idx=None):
    """Find the SwitchGLU module in a model layer, supporting multiple architectures.

    Returns (switch_mlp, key_prefix_base) or (None, None) if not an MoE layer.

    Supported paths:
      - layer.mlp.switch_mlp (Qwen, DeepSeek, GLM, Hunyuan, Jamba, OLMoE)
      - layer.block_sparse_moe.switch_mlp (Mixtral, PhiMoE, MiniMax, GraniteMoE)
    """
    prefix = f"model.layers.{layer_idx}" if layer_idx is not None else None

    if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
        switch = layer.mlp.switch_mlp
        key_base = f"{prefix}.mlp.switch_mlp" if prefix else "mlp.switch_mlp"
        return switch, key_base

    if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "switch_mlp"):
        switch = layer.block_sparse_moe.switch_mlp
        key_base = f"{prefix}.block_sparse_moe.switch_mlp" if prefix else "block_sparse_moe.switch_mlp"
        return switch, key_base

    return None, None


def _find_moe_block(layer):
    """Find the MoE block in a layer (the parent of switch_mlp).

    Returns the MoE block or None. Works for both Qwen (layer.mlp) and
    Mixtral (layer.block_sparse_moe) families.
    """
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
        return layer.mlp
    if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "switch_mlp"):
        return layer.block_sparse_moe
    return None


def _detect_num_experts(switch_mlp):
    """Detect number of experts from a SwitchGLU module."""
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(switch_mlp, name, None)
        if proj is not None and hasattr(proj, "num_experts"):
            return proj.num_experts
    return 512


# ---------------------------------------------------------------------------
# Phase 2: LCP cache (eval-based, per-token cache lookup)
# ---------------------------------------------------------------------------

class ExpertCache:
    """Per-layer LCP (Least Critical Priority) cache for expert weights.

    Shared by all 3 projections (gate/up/down) within one MoE layer.
    Eviction priority: P = μ × 0.25^(ν / 128) where μ = activation count,
    ν = steps since last activation. Lower P → evicted first.

    Step tracking: SwitchGLU calls up_proj → gate_proj → down_proj sequentially,
    so _proj_count cycles 0→1→2→0. Step increments on the first call (count == 0
    after reset), and frequency/recency are updated once per step.
    """
    __slots__ = ('entries', 'frequency', 'last_active', 'step',
                 '_proj_count', 'capacity', 'hits', 'misses',
                 'all_seen')

    def __init__(self, capacity: int):
        self.entries: dict[int, dict[str, tuple]] = {}
        self.frequency: dict[int, int] = {}
        self.last_active: dict[int, int] = {}
        self.step = 0
        self._proj_count = 0
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.all_seen: set[int] = set()

    def projection_called(self, expert_ids: np.ndarray):
        """Called once per projection. Increments step every 3rd call."""
        if self._proj_count == 0:
            self.step += 1
            for eid in expert_ids:
                eid = int(eid)
                self.frequency[eid] = self.frequency.get(eid, 0) + 1
                self.last_active[eid] = self.step
                self.all_seen.add(eid)
        self._proj_count = (self._proj_count + 1) % 3

    def lookup(self, expert_id: int, proj_name: str):
        """Return cached (w, s, b) or None. No stats tracking."""
        entry = self.entries.get(expert_id)
        if entry is not None:
            return entry.get(proj_name)
        return None

    def put(self, expert_id: int, proj_name: str, w, s, b):
        if expert_id not in self.entries:
            self.entries[expert_id] = {}
        self.entries[expert_id][proj_name] = (w, s, b)

    def evict_if_needed(self, protected: set[int]):
        """Evict lowest-priority experts until at or under capacity."""
        while len(self.entries) > self.capacity:
            worst_id = None
            worst_p = float('inf')
            for eid in self.entries:
                if eid in protected:
                    continue
                p = self._priority(eid)
                if p < worst_p:
                    worst_p = p
                    worst_id = eid
            if worst_id is None:
                break
            del self.entries[worst_id]
            del self.frequency[worst_id]
            del self.last_active[worst_id]

    def _priority(self, expert_id: int) -> float:
        mu = self.frequency.get(expert_id, 0)
        nu = self.step - self.last_active.get(expert_id, 0)
        return mu * (0.25 ** (nu / 128))


# ---------------------------------------------------------------------------
# Phase 1: Lazy loading (no cache, fresh mx.load per call)
# ---------------------------------------------------------------------------

class LazyQuantizedSwitchLinear(nn.Module):
    """Drop-in replacement for QuantizedSwitchLinear that loads experts on demand.

    On each forward call: loads a fresh lazy ref from the safetensors shard,
    slices only the needed experts, and lets the full tensor be freed after
    evaluation. This avoids the problem where evaluating a slice from a lazy
    tensor permanently materializes the full source tensor in Metal memory.
    """

    def __init__(self, shard_path: str, key_prefix: str, group_size: int,
                 bits: int, mode: str, shard_map: dict[str, str] | None = None):
        super().__init__()
        self._shard_path = shard_path
        self._key_prefix = key_prefix
        self._shard_map = shard_map
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.freeze()

    def _load_expert_subset(self, expert_ids: mx.array):
        """Load only the needed experts from the safetensors shard."""
        shard = mx.load(self._shard_path)
        return _load_proj_experts(shard, self._key_prefix, expert_ids,
                                  shard_map=self._shard_map)

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


# ---------------------------------------------------------------------------
# Phase 2: Cached loading (eval-based, LCP eviction)
# ---------------------------------------------------------------------------

class CachedQuantizedSwitchLinear(nn.Module):
    """Expert loader with per-layer LCP caching. Drop-in for QuantizedSwitchLinear.

    Cache hits serve weights from Metal memory. Misses batch-load from the
    safetensors shard, eval once, then insert individually into the cache.
    """

    def __init__(self, shard_path: str, key_prefix: str, group_size: int,
                 bits: int, mode: str, proj_name: str,
                 cache: ExpertCache,
                 shard_map: dict[str, str] | None = None):
        super().__init__()
        self._shard_path = shard_path
        self._key_prefix = key_prefix
        self._shard_map = shard_map
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._proj_name = proj_name
        self._cache = cache
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        mx.eval(indices)
        indices_np = np.asarray(indices.reshape(-1))
        unique_ids = np.unique(indices_np)

        self._cache.projection_called(unique_ids)

        # Partition into hits and misses
        hit_ids = []
        miss_ids = []
        for eid in unique_ids:
            eid = int(eid)
            if self._cache.lookup(eid, self._proj_name) is not None:
                hit_ids.append(eid)
                self._cache.hits += 1
            else:
                miss_ids.append(eid)
                self._cache.misses += 1

        # Batch-load misses from shard
        if miss_ids:
            miss_arr = mx.array(miss_ids)
            shard = mx.load(self._shard_path)
            w_batch, s_batch, b_batch = _load_proj_experts(shard, self._key_prefix, miss_arr,
                                                            shard_map=self._shard_map)
            mx.eval(w_batch, s_batch) if b_batch is None else mx.eval(w_batch, s_batch, b_batch)

            for i, eid in enumerate(miss_ids):
                self._cache.put(
                    eid, self._proj_name,
                    w_batch[i], s_batch[i],
                    b_batch[i] if b_batch is not None else None,
                )

        protected = set(int(e) for e in unique_ids)
        self._cache.evict_if_needed(protected)

        # Assemble full tensors from cache in expert order
        all_ids = sorted(int(e) for e in unique_ids)
        ws, ss, bs = [], [], []
        has_bias = None
        for eid in all_ids:
            w, s, b = self._cache.lookup(eid, self._proj_name)
            ws.append(w)
            ss.append(s)
            if has_bias is None:
                has_bias = b is not None
            if has_bias:
                bs.append(b)

        w_cat = mx.stack(ws)
        s_cat = mx.stack(ss)
        b_cat = mx.stack(bs) if has_bias else None

        # Remap global expert indices to 0..N-1 local indices
        unique_sorted = np.array(all_ids, dtype=np.int32)
        remap = np.empty(unique_sorted[-1] + 1, dtype=np.int32)
        remap[unique_sorted] = np.arange(len(unique_sorted), dtype=np.int32)
        remapped = mx.array(remap[indices_np].reshape(indices.shape))

        return mx.gather_qmm(
            x,
            w_cat,
            s_cat,
            b_cat,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


# ---------------------------------------------------------------------------
# Phase 3: Predictive cache (zero-eval forward pass)
# ---------------------------------------------------------------------------

class PredictiveExpertCache:
    """Per-layer cache with GPU-resident weight tensors and lookup table.

    Pre-loads a subset of experts into Metal memory at startup. During the
    forward pass, a lookup table remaps global expert IDs (0-511) to cache
    slots (0 to C-1) entirely on GPU — no mx.eval needed. Uncached experts
    map to slot 0 (fallback).

    Supports dynamic updates between tokens: captures router indices during
    forward pass, then swaps cold experts for newly-requested ones.
    """
    __slots__ = ('capacity', 'num_experts', 'lookup',
                 'weights', 'scales', 'biases',
                 'cached_ids', 'cached_set',
                 'frequency', 'last_active', 'step',
                 '_indices_buffer',
                 '_shard_paths', '_key_prefixes', '_shard_map',
                 'total_requests', 'total_fallbacks',
                 'pinned_set')

    def __init__(self, capacity: int, num_experts: int = 512):
        self.capacity = capacity
        self.num_experts = num_experts
        self.weights: dict[str, mx.array] = {}
        self.scales: dict[str, mx.array] = {}
        self.biases: dict[str, mx.array | None] = {}
        self.lookup: mx.array | None = None
        self.cached_ids: list[int] = []
        self.cached_set: set[int] = set()
        self.frequency: dict[int, int] = {}
        self.last_active: dict[int, int] = {}
        self.step: int = 0
        self._indices_buffer: list[mx.array] = []
        self._shard_paths: dict[str, str] = {}
        self._key_prefixes: dict[str, str] = {}
        self._shard_map: dict[str, str] | None = None
        self.total_requests: int = 0
        self.total_fallbacks: int = 0
        self.pinned_set: set[int] = set()

    def build_lookup(self, cached_ids: list[int]):
        """Build GPU-resident lookup table. Uncached IDs map to slot 0."""
        self.cached_ids = list(cached_ids)
        self.cached_set = set(cached_ids)
        for eid in cached_ids:
            self.frequency.setdefault(eid, 1)
            self.last_active.setdefault(eid, 0)
        lookup_np = np.zeros(self.num_experts, dtype=np.int32)
        for slot, eid in enumerate(cached_ids):
            lookup_np[eid] = slot
        self.lookup = mx.array(lookup_np)

    def remap(self, indices: mx.array) -> mx.array:
        """Map global expert IDs to cache slots. Pure mx.array op, no eval."""
        return self.lookup[indices]

    def _lcp_priority(self, eid: int) -> float:
        mu = self.frequency.get(eid, 0)
        nu = self.step - self.last_active.get(eid, 0)
        return mu * (0.25 ** (nu / 128))

    def update(self) -> dict:
        """Process buffered indices and swap cold experts for missed ones.

        Call between tokens. Skips the last buffered entry (in-flight due
        to async_eval double-buffering). Returns stats dict.
        """
        if len(self._indices_buffer) < 2:
            return {"swaps": 0, "fallbacks": 0, "requests": 0}

        to_process = self._indices_buffer[:-1]
        self._indices_buffer = self._indices_buffer[-1:]

        all_requested: set[int] = set()
        for indices in to_process:
            flat = np.asarray(indices.reshape(-1))
            unique = set(int(x) for x in np.unique(flat))
            all_requested |= unique

        self.step += 1
        for eid in all_requested:
            self.frequency[eid] = self.frequency.get(eid, 0) + 1
            self.last_active[eid] = self.step

        misses = all_requested - self.cached_set
        n_requests = len(all_requested)
        n_fallbacks = len(misses)
        self.total_requests += n_requests
        self.total_fallbacks += n_fallbacks

        if not misses or not self._shard_paths:
            return {"swaps": 0, "fallbacks": n_fallbacks, "requests": n_requests}

        # Find coldest cached experts to evict (exclude requested and pinned)
        evict_candidates = [
            (self._lcp_priority(eid), slot, eid)
            for slot, eid in enumerate(self.cached_ids)
            if eid not in all_requested and eid not in self.pinned_set
        ]
        evict_candidates.sort()

        swaps: list[tuple[int, int, int]] = []  # (slot, old_eid, new_eid)
        miss_list = sorted(misses)
        for new_eid in miss_list:
            if not evict_candidates:
                break
            _, slot, old_eid = evict_candidates.pop(0)
            swaps.append((slot, old_eid, new_eid))

        if not swaps:
            return {"swaps": 0, "fallbacks": n_fallbacks, "requests": n_requests}

        # Cap swaps per layer to bound transient memory from shard materialization.
        # Each swap temporarily materializes ~336 MB (full source tensor).
        # Remaining misses get picked up on subsequent tokens.
        MAX_SWAPS = 10
        swaps = swaps[:MAX_SWAPS]

        # Load new experts and scatter into stacked tensors
        new_eids = mx.array([new_eid for _, _, new_eid in swaps])
        slot_indices = mx.array([slot for slot, _, _ in swaps])
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            shard_path = self._shard_paths[proj_name]
            key_prefix = self._key_prefixes[proj_name]
            shard = mx.load(shard_path)
            new_w, new_s, new_b = _load_proj_experts(shard, key_prefix, new_eids,
                                                      shard_map=self._shard_map)
            del shard

            if new_b is None:
                mx.eval(new_w, new_s)
            else:
                mx.eval(new_w, new_s, new_b)

            w = self.weights.pop(proj_name)
            w[slot_indices] = new_w
            self.weights[proj_name] = w

            s = self.scales.pop(proj_name)
            s[slot_indices] = new_s
            self.scales[proj_name] = s

            if self.biases[proj_name] is not None and new_b is not None:
                b = self.biases.pop(proj_name)
                b[slot_indices] = new_b
                self.biases[proj_name] = b

        mx.clear_cache()

        # Update cached_ids and lookup table
        for slot, old_eid, new_eid in swaps:
            self.cached_set.discard(old_eid)
            self.cached_set.add(new_eid)
            self.cached_ids[slot] = new_eid
            self.frequency.pop(old_eid, None)
            self.last_active.pop(old_eid, None)

        lookup_np = np.zeros(self.num_experts, dtype=np.int32)
        for slot, eid in enumerate(self.cached_ids):
            lookup_np[eid] = slot
        self.lookup = mx.array(lookup_np)
        mx.eval(self.lookup)

        return {"swaps": len(swaps), "fallbacks": n_fallbacks, "requests": n_requests}


class PredictiveCachedSwitchLinear(nn.Module):
    """Zero-eval expert dispatch using pre-loaded weights and GPU lookup table.

    The forward pass stays entirely lazy — indices are remapped via a
    pre-built lookup table on GPU, and gather_qmm uses pre-loaded weight
    tensors already in Metal memory. No mx.eval until the output token.

    Captures router indices for dynamic cache updates between tokens.
    Only the first projection (up_proj, called first by SwitchGLU) captures
    indices to avoid triple-buffering the same data.
    """

    def __init__(self, group_size: int, bits: int, mode: str,
                 proj_name: str, cache: PredictiveExpertCache):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._proj_name = proj_name
        self._cache = cache
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        if self._proj_name == "up_proj":
            self._cache._indices_buffer.append(indices)
        local_indices = self._cache.remap(indices)
        return mx.gather_qmm(
            x,
            self._cache.weights[self._proj_name],
            self._cache.scales[self._proj_name],
            self._cache.biases[self._proj_name],
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


class SyncPredictiveCachedSwitchLinear(nn.Module):
    """Same as PredictiveCachedSwitchLinear but WITH mx.eval(indices).

    Isolates the sync-point hypothesis: pre-stacked tensors, GPU lookup table,
    but forces a per-layer pipeline flush via mx.eval. Comparing this against
    PredictiveCachedSwitchLinear measures the cost of sync points alone.
    """

    def __init__(self, group_size: int, bits: int, mode: str,
                 proj_name: str, cache: PredictiveExpertCache):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._proj_name = proj_name
        self._cache = cache
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        if self._proj_name == "up_proj":
            self._cache._indices_buffer.append(indices)
        mx.eval(indices)
        local_indices = self._cache.remap(indices)
        return mx.gather_qmm(
            x,
            self._cache.weights[self._proj_name],
            self._cache.scales[self._proj_name],
            self._cache.biases[self._proj_name],
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------

def _build_shard_map(model_path: Path) -> dict[str, str]:
    """Read model.safetensors.index.json and return {key: absolute_shard_path}.

    For models with per-expert safetensors keys (e.g. Mixtral, GLM), adds
    synthetic stacked-format keys so callers can look up
    ``{prefix}.switch_mlp.gate_proj.weight`` even though the actual file
    stores ``{prefix}.experts.0.w1.weight``.  The shard path returned is
    expert 0's shard (all experts for a layer share a shard).
    """
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_map = {key: str(model_path / shard) for key, shard in weight_map.items()}

    # Detect per-expert format and add synthetic stacked keys.
    # Two naming conventions:
    #   Mixtral:  {prefix}.experts.0.w1.weight  (w1->gate, w2->down, w3->up)
    #   GLM/DS:   {prefix}.experts.0.gate_proj.weight
    _w_to_proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    seen_expert_prefixes: set[str] = set()
    for key in weight_map:
        if ".experts.0." not in key:
            continue
        # e.g. "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        # Split into prefix, "experts", "0", sub_name, weight_type
        idx = key.index(".experts.0.")
        moe_prefix = key[:idx]  # "model.layers.0.block_sparse_moe"
        remainder = key[idx + len(".experts.0."):]  # "w1.weight"
        parts = remainder.split(".")
        if len(parts) != 2:
            continue
        sub_name, wt = parts  # ("w1", "weight")
        proj_name = _w_to_proj.get(sub_name, sub_name)
        synth_key = f"{moe_prefix}.switch_mlp.{proj_name}.{wt}"
        if synth_key not in shard_map:
            shard_map[synth_key] = str(model_path / weight_map[key])
            seen_expert_prefixes.add(moe_prefix)

    return shard_map


# Mapping from canonical projection names to per-expert sub-names.
_PROJ_TO_EXPERT_NAMES = {
    "gate_proj": ("gate_proj", "w1"),
    "up_proj": ("up_proj", "w3"),
    "down_proj": ("down_proj", "w2"),
}


def _load_proj_experts(shard: dict, key_prefix: str, expert_ids,
                       shard_map: dict[str, str] | None = None,
                       ) -> tuple[mx.array, mx.array, mx.array | None]:
    """Load weight/scales/biases for ``expert_ids`` from a safetensors shard.

    Handles both:
      - **Stacked format** (Qwen): ``{key_prefix}.weight`` is a (E, ...) tensor.
      - **Per-expert format** (Mixtral, GLM): individual keys like
        ``{moe_base}.experts.{e}.{sub}.weight``.

    For per-expert format, some experts may live in a different shard file.
    Pass ``shard_map`` (from ``_build_shard_map``) to enable cross-shard
    loading.  Extra shards are loaded on demand and freed immediately.
    """
    stacked_key = f"{key_prefix}.weight"
    if stacked_key in shard:
        w = shard[stacked_key][expert_ids]
        s = shard[f"{key_prefix}.scales"][expert_ids]
        biases_key = f"{key_prefix}.biases"
        b = shard[biases_key][expert_ids] if biases_key in shard else None
        return w, s, b

    # Per-expert format: key_prefix is e.g.
    #   "model.layers.0.block_sparse_moe.switch_mlp.gate_proj"
    # We need to map back to "model.layers.0.block_sparse_moe.experts.{e}.w1"
    parts = key_prefix.rsplit(".", 1)  # ("...switch_mlp", "gate_proj")
    switch_prefix = parts[0]  # "...switch_mlp"
    proj_name = parts[1]  # "gate_proj"
    moe_base = switch_prefix.rsplit(".switch_mlp", 1)[0]  # "...block_sparse_moe"

    candidates = _PROJ_TO_EXPERT_NAMES.get(proj_name, (proj_name,))

    ids = np.asarray(expert_ids).reshape(-1) if not isinstance(expert_ids, np.ndarray) else expert_ids.reshape(-1)

    # Cache for extra shards loaded on demand (path -> lazy dict)
    _extra_shards: dict[str, dict] = {}

    def _resolve_shard(expert_key: str) -> dict:
        """Return the shard dict containing ``expert_key``."""
        if expert_key in shard:
            return shard
        if shard_map is None:
            return shard  # caller didn't provide map; fall through to KeyError
        alt_path = shard_map.get(expert_key)
        if alt_path is None:
            return shard
        if alt_path not in _extra_shards:
            _extra_shards[alt_path] = mx.load(alt_path)
        return _extra_shards[alt_path]

    ws, ss, bs = [], [], []
    has_bias = None
    for eid in ids:
        eid = int(eid)
        loaded = False
        for sub in candidates:
            expert_key = f"{moe_base}.experts.{eid}.{sub}.weight"
            s_dict = _resolve_shard(expert_key)
            if expert_key in s_dict:
                ws.append(s_dict[expert_key])
                ss.append(s_dict[f"{moe_base}.experts.{eid}.{sub}.scales"])
                b_key = f"{moe_base}.experts.{eid}.{sub}.biases"
                if has_bias is None:
                    has_bias = b_key in s_dict
                if has_bias:
                    bs.append(s_dict[b_key])
                loaded = True
                break
        if not loaded:
            raise KeyError(
                f"No expert key found for expert {eid}, "
                f"tried: {[f'{moe_base}.experts.{eid}.{s}.weight' for s in candidates]}"
            )

    w = mx.stack(ws)
    s = mx.stack(ss)
    b = mx.stack(bs) if has_bias else None
    return w, s, b


def enable_lazy_experts(model, model_path: Path, cache_capacity_per_layer: int = 0,
                        predictive: bool = False) -> int:
    """Replace QuantizedSwitchLinear modules in MoE layers with lazy/cached versions.

    Args:
        model: The loaded MLX model (with lazy=True).
        model_path: Path to the model directory containing safetensors shards.
        cache_capacity_per_layer: Number of experts to cache per layer. 0 = no cache
            (Phase 1 lazy loading). > 0 = LCP-cached or predictive loading.
        predictive: If True and cache_capacity > 0, use zero-eval predictive cache
            (Phase 3). Pre-loads experts at startup, eliminates per-layer mx.eval.

    Returns:
        Number of modules replaced (expected: 48 layers x 3 = 144).
    """
    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)

    if predictive and cache_capacity_per_layer > 0:
        return _enable_predictive(model, shard_map, cache_capacity_per_layer)
    elif cache_capacity_per_layer > 0:
        return _enable_cached(model, shard_map, cache_capacity_per_layer)
    else:
        return _enable_lazy(model, shard_map)


def _enable_lazy(model, shard_map: dict, ) -> int:
    replaced = 0
    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        for name in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(switch, name)
            if not isinstance(orig, QuantizedSwitchLinear):
                continue
            key_prefix = f"{key_base}.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            replacement = LazyQuantizedSwitchLinear(
                shard_path=shard_path,
                key_prefix=key_prefix,
                group_size=orig.group_size,
                bits=orig.bits,
                mode=orig.mode,
                shard_map=shard_map,
            )
            setattr(switch, name, replacement)
            replaced += 1
    return replaced


def _enable_cached(model, shard_map: dict, capacity: int) -> int:
    replaced = 0
    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        layer_cache = ExpertCache(capacity)
        for name in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(switch, name)
            if not isinstance(orig, QuantizedSwitchLinear):
                continue
            key_prefix = f"{key_base}.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            replacement = CachedQuantizedSwitchLinear(
                shard_path=shard_path,
                key_prefix=key_prefix,
                group_size=orig.group_size,
                bits=orig.bits,
                mode=orig.mode,
                proj_name=name,
                cache=layer_cache,
                shard_map=shard_map,
            )
            setattr(switch, name, replacement)
            replaced += 1
    return replaced


def _enable_predictive(model, shard_map: dict, capacity: int) -> int:
    """Install Phase 2 modules for warmup. Call upgrade_to_predictive() after."""
    return _enable_cached(model, shard_map, capacity)


def reset_to_cached(model, model_path: Path, capacity: int) -> int:
    """Downgrade predictive modules back to Phase 2 cached for re-warmup.

    Replaces PredictiveCachedSwitchLinear with fresh CachedQuantizedSwitchLinear,
    freeing the pre-stacked expert tensors. Use this to re-warm on a new prompt
    without reloading the entire model.
    """
    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)

    reset = 0
    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        first = getattr(switch, "gate_proj")
        if not isinstance(first, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue

        layer_cache = ExpertCache(capacity)
        for name in ("gate_proj", "up_proj", "down_proj"):
            pred_mod = getattr(switch, name)
            key_prefix = f"{key_base}.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            replacement = CachedQuantizedSwitchLinear(
                shard_path=shard_path,
                key_prefix=key_prefix,
                group_size=pred_mod.group_size,
                bits=pred_mod.bits,
                mode=pred_mod.mode,
                proj_name=name,
                cache=layer_cache,
                shard_map=shard_map,
            )
            setattr(switch, name, replacement)
            reset += 1

    mx.clear_cache()
    return reset


def upgrade_to_predictive(model, model_path: Path, capacity,
                          sync: bool = False) -> int:
    """Harvest Phase 2 LCP caches into zero-eval predictive tensors.

    Batched shard loading: groups expert loads by safetensors shard file and
    loads each shard only once (9 loads instead of 144). Evals per-layer within
    each shard batch to control memory.

    Args:
        capacity: int for uniform capacity, or list[int] for per-MoE-layer capacities.
        sync: If True, use SyncPredictiveCachedSwitchLinear (adds mx.eval per layer).

    Returns number of modules upgraded.
    """
    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)
    per_layer_caps = isinstance(capacity, (list, tuple))

    # --- Pass 1: harvest LCP caches, determine what needs disk loading ---
    # layer_meta[i] = {cached_ids, pred_cache, harvested[proj][(slot,w,s,b)],
    #                  to_load[proj][(slot,eid)], has_bias, phase2_mods, filler_count, disc_count}
    layer_meta = {}
    moe_idx = 0

    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        first_proj = getattr(switch, "gate_proj")
        if not isinstance(first_proj, CachedQuantizedSwitchLinear):
            continue

        lcp_cache = first_proj._cache
        num_experts = _detect_num_experts(switch)
        C = min(capacity[moe_idx] if per_layer_caps else capacity, num_experts)
        moe_idx += 1

        discovered = sorted(
            lcp_cache.entries.keys(),
            key=lambda eid: lcp_cache._priority(eid),
            reverse=True,
        )[:C]
        discovered_set = set(discovered)

        filler = []
        for eid in range(num_experts):
            if len(discovered) + len(filler) >= C:
                break
            if eid not in discovered_set:
                filler.append(eid)
        cached_ids = list(discovered) + filler

        pred_cache = PredictiveExpertCache(C, num_experts)
        harvested = {}
        to_load = {}
        has_bias = None
        phase2_mods = {}

        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mods[name] = getattr(switch, name)
            h_list = []
            load_list = []
            for slot, eid in enumerate(cached_ids):
                cached = lcp_cache.lookup(eid, name)
                if cached is not None:
                    w, s, b = cached
                    if has_bias is None:
                        has_bias = b is not None
                    h_list.append((slot, w, s, b))
                else:
                    load_list.append((slot, eid))
            harvested[name] = h_list
            to_load[name] = load_list

        layer_meta[i] = {
            "cached_ids": cached_ids,
            "pred_cache": pred_cache,
            "harvested": harvested,
            "to_load": to_load,
            "has_bias": has_bias if has_bias is not None else True,
            "phase2_mods": phase2_mods,
            "disc_count": len(discovered),
            "filler_count": len(filler),
            "C": C,
            "lcp_cache": lcp_cache,
            "key_base": key_base,
        }

    # --- Pass 2: group disk loads by shard, load each shard once ---
    # Build: shard_path -> [(layer_i, proj_name, key_prefix, [(slot, eid)])]
    shard_groups: dict[str, list[tuple]] = {}
    for i, meta in layer_meta.items():
        for name in ("gate_proj", "up_proj", "down_proj"):
            if not meta["to_load"][name]:
                continue
            key_prefix = f"{meta['key_base']}.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            shard_groups.setdefault(shard_path, []).append(
                (i, name, key_prefix, meta["to_load"][name]))

    # loaded[i][name] = {slot: (w, s, b)} for disk-loaded experts
    loaded: dict[int, dict[str, dict[int, tuple]]] = {}

    for shard_path, group in shard_groups.items():
        shard = mx.load(shard_path)

        # Extract slices for all layers/projections in this shard.
        # Eval per-layer to bound memory (don't accumulate all layers' data).
        layers_in_batch = sorted(set(layer_i for layer_i, _, _, _ in group))
        for layer_i in layers_in_batch:
            layer_entries = [(n, kp, slots) for li, n, kp, slots in group if li == layer_i]
            to_eval = []
            for name, key_prefix, slot_eids in layer_entries:
                load_ids = mx.array([eid for _, eid in slot_eids])
                w_batch, s_batch, b_batch = _load_proj_experts(shard, key_prefix, load_ids,
                                                              shard_map=shard_map)
                to_eval.extend([w_batch, s_batch])
                if b_batch is not None:
                    to_eval.append(b_batch)

                slot_map = {}
                for j, (slot, _) in enumerate(slot_eids):
                    slot_map[slot] = (w_batch[j], s_batch[j],
                                      b_batch[j] if b_batch is not None else None)
                loaded.setdefault(layer_i, {})[name] = slot_map

            mx.eval(*to_eval)

        del shard

    # --- Pass 3: assemble stacked tensors, build lookups, install modules ---
    upgraded = 0
    cls = SyncPredictiveCachedSwitchLinear if sync else PredictiveCachedSwitchLinear

    for i, meta in layer_meta.items():
        pred_cache = meta["pred_cache"]
        cached_ids = meta["cached_ids"]
        has_bias = meta["has_bias"]
        C = meta["C"]

        for name in ("gate_proj", "up_proj", "down_proj"):
            ws, ss, bs = [], [], []
            # Merge harvested (from LCP) and loaded (from disk) into slot order
            harvested_map = {slot: (w, s, b) for slot, w, s, b in meta["harvested"][name]}
            loaded_map = loaded.get(i, {}).get(name, {})

            for slot in range(C):
                if slot in harvested_map:
                    w, s, b = harvested_map[slot]
                else:
                    w, s, b = loaded_map[slot]
                ws.append(w)
                ss.append(s)
                if has_bias:
                    bs.append(b)

            pred_cache.weights[name] = mx.stack(ws)
            pred_cache.scales[name] = mx.stack(ss)
            pred_cache.biases[name] = mx.stack(bs) if has_bias else None

        key_base = meta["key_base"]
        for name in ("gate_proj", "up_proj", "down_proj"):
            key_prefix = f"{key_base}.{name}"
            pred_cache._shard_paths[name] = shard_map[f"{key_prefix}.weight"]
            pred_cache._key_prefixes[name] = key_prefix
        pred_cache._shard_map = shard_map

        pred_cache.build_lookup(cached_ids)
        mx.eval(pred_cache.lookup)

        switch, _ = _find_switch_mlp(model.layers[i], i)
        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mod = meta["phase2_mods"][name]
            replacement = cls(
                group_size=phase2_mod.group_size,
                bits=phase2_mod.bits,
                mode=phase2_mod.mode,
                proj_name=name,
                cache=pred_cache,
            )
            setattr(switch, name, replacement)
            upgraded += 1

        meta["lcp_cache"].entries.clear()
        meta["lcp_cache"].frequency.clear()
        meta["lcp_cache"].last_active.clear()

        print(f"  Layer {i}: {meta['disc_count']} discovered + {meta['filler_count']} filler "
              f"= {C} experts ({mx.get_active_memory() / 1e9:.1f} GB)")

    return upgraded


# ---------------------------------------------------------------------------
# Dynamic cache management
# ---------------------------------------------------------------------------

def dynamic_cache_update(model, max_layer_updates: int = 12) -> list[dict]:
    """Process buffered router indices and swap cold experts for missed ones.

    Call between tokens during generation. Handles the async_eval
    double-buffering by skipping in-flight indices.

    Args:
        max_layer_updates: Max layers to perform swaps on per call. Limits
            transient memory from shard loading. Other layers still track
            stats but defer swaps to later calls.

    Returns per-layer stats: [{"layer": i, "swaps": n, "fallbacks": n, "requests": n}, ...]
    """
    stats = []
    swap_budget = max_layer_updates
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue
        cache = proj._cache

        # Always process indices for stats tracking, but only do swaps
        # if we have budget remaining
        if swap_budget > 0:
            layer_stats = cache.update()
            if layer_stats["swaps"] > 0:
                swap_budget -= 1
        else:
            # Drain indices buffer for stats only (no swaps)
            if len(cache._indices_buffer) < 2:
                layer_stats = {"swaps": 0, "fallbacks": 0, "requests": 0}
            else:
                to_process = cache._indices_buffer[:-1]
                cache._indices_buffer = cache._indices_buffer[-1:]
                all_requested: set[int] = set()
                for indices in to_process:
                    flat = np.asarray(indices.reshape(-1))
                    all_requested |= set(int(x) for x in np.unique(flat))
                cache.step += 1
                for eid in all_requested:
                    cache.frequency[eid] = cache.frequency.get(eid, 0) + 1
                    cache.last_active[eid] = cache.step
                misses = all_requested - cache.cached_set
                cache.total_requests += len(all_requested)
                cache.total_fallbacks += len(misses)
                layer_stats = {"swaps": 0, "fallbacks": len(misses), "requests": len(all_requested)}

        layer_stats["layer"] = i
        stats.append(layer_stats)
    return stats


def get_fallback_stats(model) -> dict:
    """Collect cumulative fallback stats from all PredictiveExpertCache instances."""
    total_requests = 0
    total_fallbacks = 0
    layer_stats = []

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue
        cache = proj._cache
        total_requests += cache.total_requests
        total_fallbacks += cache.total_fallbacks
        if cache.total_requests > 0:
            layer_stats.append({
                "layer": i,
                "requests": cache.total_requests,
                "fallbacks": cache.total_fallbacks,
                "fallback_rate": cache.total_fallbacks / cache.total_requests,
                "cached_experts": len(cache.cached_set),
            })

    return {
        "total_requests": total_requests,
        "total_fallbacks": total_fallbacks,
        "fallback_rate": total_fallbacks / total_requests if total_requests > 0 else 0.0,
        "layers": layer_stats,
    }


# ---------------------------------------------------------------------------
# Delta warmup (fast multi-turn cache update)
# ---------------------------------------------------------------------------

def delta_warmup(model, tokenizer, model_path, new_prompt, discovery_tokens=10):
    """Fast cache update for multi-turn: discover missing experts, swap them in.

    Instead of the slow reset → re-warmup → upgrade cycle (~70s), keeps the
    existing predictive cache and does a fast "delta discovery" pass:
    1. Run discovery_tokens through predictive cache (~0.5s at 20 tok/s)
    2. Collect which experts were requested but not cached
    3. Evict cold experts, load missing ones from disk
    4. One-time tensor rebuild per affected layer

    Returns dict with timing breakdown and per-layer swap stats.
    """
    import mlx_lm as _mlx_lm
    import time

    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)

    # Clear stale indices from previous generation
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache._indices_buffer.clear()

    # Step 1: Discovery pass at full speed through existing predictive cache
    t0 = time.perf_counter()
    _mlx_lm.generate(model, tokenizer, prompt=new_prompt,
                     max_tokens=discovery_tokens, verbose=False)
    t_discovery = time.perf_counter() - t0

    # Step 2: Collect requested expert IDs and identify misses per layer
    t1 = time.perf_counter()
    layer_info = {}

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue

        cache = proj._cache

        # Drain all buffered indices (generation complete, all evaluated)
        all_requested = set()
        for indices in cache._indices_buffer:
            flat = np.asarray(indices.reshape(-1))
            all_requested |= set(int(x) for x in np.unique(flat))
        cache._indices_buffer.clear()

        missing = all_requested - cache.cached_set

        # Cold: cached but not requested, sorted by LCP priority (lowest first)
        cold = sorted(
            [(cache._lcp_priority(eid), slot, eid)
             for slot, eid in enumerate(cache.cached_ids)
             if eid not in all_requested],
        )

        layer_info[i] = {
            "requested": all_requested,
            "missing": missing,
            "cold": cold,
            "cache": cache,
        }

    # Step 3: Compute per-layer swaps and group shard loads across all layers
    total_swaps = 0
    total_missing = 0
    per_layer_stats = []
    # layer_swaps[i] = [(slot, old_eid, new_eid), ...]
    layer_swaps: dict[int, list[tuple]] = {}

    for i, info in layer_info.items():
        missing = info["missing"]
        cold = list(info["cold"])
        total_missing += len(missing)

        if not missing:
            per_layer_stats.append({"layer": i, "missing": 0, "swapped": 0})
            continue

        swaps = []
        for new_eid in sorted(missing):
            if not cold:
                break
            _, slot, old_eid = cold.pop(0)
            swaps.append((slot, old_eid, new_eid))

        if not swaps:
            per_layer_stats.append({"layer": i, "missing": len(missing), "swapped": 0})
            continue

        layer_swaps[i] = swaps
        total_swaps += len(swaps)
        per_layer_stats.append({
            "layer": i, "missing": len(missing), "swapped": len(swaps),
        })

    # Group all shard loads across layers:
    # shard_path -> [(layer_i, proj_name, key_prefix, new_eids_array)]
    shard_groups: dict[str, list[tuple]] = {}
    for i, swaps in layer_swaps.items():
        cache = layer_info[i]["cache"]
        new_eids = mx.array([new_eid for _, _, new_eid in swaps])
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            sp = cache._shard_paths[proj_name]
            kp = cache._key_prefixes[proj_name]
            shard_groups.setdefault(sp, []).append((i, proj_name, kp, new_eids))

    # loaded_experts[i][proj_name] = (new_w, new_s, new_b)
    loaded_experts: dict[int, dict[str, tuple]] = {}

    for shard_path, group in shard_groups.items():
        shard = mx.load(shard_path)

        # Extract and eval per-layer within this shard to bound memory
        layers_in_batch = sorted(set(li for li, _, _, _ in group))
        for layer_i in layers_in_batch:
            layer_entries = [(pn, kp, eids) for li, pn, kp, eids in group if li == layer_i]
            to_eval = []
            for proj_name, key_prefix, new_eids in layer_entries:
                new_w, new_s, new_b = _load_proj_experts(shard, key_prefix, new_eids,
                                                          shard_map=shard_map)
                loaded_experts.setdefault(layer_i, {})[proj_name] = (new_w, new_s, new_b)
                to_eval.extend([new_w, new_s])
                if new_b is not None:
                    to_eval.append(new_b)
            mx.eval(*to_eval)

        del shard

    # Step 4: Scatter-update per layer and rebuild lookups
    for i, swaps in layer_swaps.items():
        cache = layer_info[i]["cache"]
        slot_indices = mx.array([slot for slot, _, _ in swaps])

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            new_w, new_s, new_b = loaded_experts[i][proj_name]

            w = cache.weights.pop(proj_name)
            w[slot_indices] = new_w
            cache.weights[proj_name] = w

            s = cache.scales.pop(proj_name)
            s[slot_indices] = new_s
            cache.scales[proj_name] = s

            if cache.biases[proj_name] is not None and new_b is not None:
                b = cache.biases.pop(proj_name)
                b[slot_indices] = new_b
                cache.biases[proj_name] = b

        del loaded_experts[i]

        for slot, old_eid, new_eid in swaps:
            cache.cached_set.discard(old_eid)
            cache.cached_set.add(new_eid)
            cache.cached_ids[slot] = new_eid
            cache.frequency.pop(old_eid, None)
            cache.last_active.pop(old_eid, None)

        lookup_np = np.zeros(cache.num_experts, dtype=np.int32)
        for slot, eid in enumerate(cache.cached_ids):
            lookup_np[eid] = slot
        cache.lookup = mx.array(lookup_np)

        # Eval per-layer to free old tensor refs (prevents OOM from accumulation)
        to_eval = [cache.lookup]
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            to_eval.append(cache.weights[proj_name])
            to_eval.append(cache.scales[proj_name])
            if cache.biases[proj_name] is not None:
                to_eval.append(cache.biases[proj_name])
        mx.eval(*to_eval)

    t_rebuild = time.perf_counter() - t1

    # Reset fallback counters for clean generation stats
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache.total_requests = 0
            proj._cache.total_fallbacks = 0
            proj._cache._indices_buffer.clear()

    mx.clear_cache()

    return {
        "discovery_time": t_discovery,
        "rebuild_time": t_rebuild,
        "total_time": t_discovery + t_rebuild,
        "total_swaps": total_swaps,
        "total_missing": total_missing,
        "per_layer": per_layer_stats,
    }


def fast_delta_warmup(model, tokenizer, model_path, new_prompt,
                      discovery_tokens=10, discovery_method="predictive",
                      min_swaps_threshold=0):
    """Optimized delta warmup with pluggable discovery and fused load+scatter.

    Two discovery methods:
      "predictive" — generate through existing predictive cache (~1.3s, high fidelity)
      "router-only" — skip MoE expert computation, run routers only (~0.5s, lower fidelity)

    Args:
      min_swaps_threshold: Skip layers with fewer missing experts than this.
          Reduces Metal buffer traffic at cost of slightly higher fallback rate.

    Returns dict with timing breakdown and per-layer swap stats.
    """
    import time

    model_path = Path(model_path)

    # Clear stale indices from previous generation
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache._indices_buffer.clear()

    # Step 1: Discovery (under cache_limit(0) to reclaim Metal headroom)
    t0 = time.perf_counter()

    with _with_cache_limit_zero():
        if discovery_method == "router-only":
            discovered = router_only_forward(model, tokenizer, new_prompt,
                                             max_tokens=discovery_tokens)
        else:
            import mlx_lm as _mlx_lm
            _mlx_lm.generate(model, tokenizer, prompt=new_prompt,
                             max_tokens=discovery_tokens, verbose=False)
            # Drain indices buffers into per-layer expert sets
            discovered = {}
            for i, layer in enumerate(model.layers):
                switch, _ = _find_switch_mlp(layer, i)
                if switch is None:
                    continue
                proj = getattr(switch, "up_proj", None)
                if not isinstance(proj, (PredictiveCachedSwitchLinear,
                                         SyncPredictiveCachedSwitchLinear)):
                    continue
                cache = proj._cache
                requested = set()
                for indices in cache._indices_buffer:
                    flat = np.asarray(indices.reshape(-1))
                    requested |= set(int(x) for x in np.unique(flat))
                cache._indices_buffer.clear()
                discovered[i] = requested

    t_discovery = time.perf_counter() - t0

    # Memory pressure check: reduce swap throughput if close to device limit
    device_mem = mx.metal.device_info()["memory_size"]
    active_mem = mx.metal.get_active_memory()
    memory_pressure = active_mem > 0.85 * device_mem
    MAX_SWAPS_PER_LAYER = 3 if memory_pressure else 10
    if memory_pressure:
        print(f"  [memory pressure: {active_mem / 1e9:.1f}/{device_mem / 1e9:.0f} GB — "
              f"limiting to {MAX_SWAPS_PER_LAYER} swaps/layer]")

    # Step 2: Compute delta (missing experts per layer, cold slots to evict)
    t1 = time.perf_counter()
    total_swaps = 0
    total_missing = 0
    per_layer_stats = []
    layer_swaps: dict[int, list[tuple]] = {}
    layer_caches: dict[int, PredictiveExpertCache] = {}

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear,
                                 SyncPredictiveCachedSwitchLinear)):
            continue

        cache = proj._cache
        layer_caches[i] = cache
        requested = discovered.get(i, set())
        missing = requested - cache.cached_set
        total_missing += len(missing)

        if not missing or len(missing) < min_swaps_threshold:
            per_layer_stats.append({"layer": i, "missing": len(missing), "swapped": 0})
            continue

        cold = sorted(
            [(cache._lcp_priority(eid), slot, eid)
             for slot, eid in enumerate(cache.cached_ids)
             if eid not in requested],
        )

        swaps = []
        for new_eid in sorted(missing):
            if not cold:
                break
            _, slot, old_eid = cold.pop(0)
            swaps.append((slot, old_eid, new_eid))

        swaps = swaps[:MAX_SWAPS_PER_LAYER]

        if not swaps:
            per_layer_stats.append({"layer": i, "missing": len(missing), "swapped": 0})
            continue

        layer_swaps[i] = swaps
        total_swaps += len(swaps)
        per_layer_stats.append({
            "layer": i, "missing": len(missing), "swapped": len(swaps),
        })

    # Step 3: Group shard loads, then fused load+scatter per layer within each shard
    # shard_path -> [layer_i, ...] (layers that need swaps in this shard)
    shard_layers: dict[str, set[int]] = {}
    for i, swaps in layer_swaps.items():
        cache = layer_caches[i]
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            sp = cache._shard_paths[proj_name]
            shard_layers.setdefault(sp, set()).add(i)

    t_shard_load = 0.0
    t_scatter = 0.0

    for shard_path, layer_set in shard_layers.items():
        t_load_start = time.perf_counter()
        shard = mx.load(shard_path)
        t_shard_load += time.perf_counter() - t_load_start

        for layer_i in sorted(layer_set):
            swaps = layer_swaps[layer_i]
            cache = layer_caches[layer_i]
            new_eids = mx.array([new_eid for _, _, new_eid in swaps])
            slot_indices = mx.array([slot for slot, _, _ in swaps])

            # Load expert data for all 3 projections from this shard
            t_load_start = time.perf_counter()
            loaded = {}
            to_eval = []
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                key_prefix = cache._key_prefixes[proj_name]
                if cache._shard_paths[proj_name] != shard_path:
                    continue
                new_w, new_s, new_b = _load_proj_experts(shard, key_prefix, new_eids,
                                                          shard_map=cache._shard_map)
                loaded[proj_name] = (new_w, new_s, new_b)
                to_eval.extend([new_w, new_s])
                if new_b is not None:
                    to_eval.append(new_b)
            mx.eval(*to_eval)
            t_shard_load += time.perf_counter() - t_load_start

            # Scatter update for loaded projections (graph construction only,
            # eval deferred to lookup rebuild section)
            t_scatter_start = time.perf_counter()
            _use_scatter_slots = hasattr(mx.fast, "scatter_slots")
            if _use_scatter_slots:
                targets = []
                values_list = []
                proj_order = []
                for proj_name, (new_w, new_s, new_b) in loaded.items():
                    targets.append(cache.weights.pop(proj_name))
                    values_list.append(new_w)
                    proj_order.append((proj_name, "weights"))

                    targets.append(cache.scales.pop(proj_name))
                    values_list.append(new_s)
                    proj_order.append((proj_name, "scales"))

                    if cache.biases[proj_name] is not None and new_b is not None:
                        targets.append(cache.biases.pop(proj_name))
                        values_list.append(new_b)
                        proj_order.append((proj_name, "biases"))

                results = mx.fast.scatter_slots(targets, slot_indices, values_list)
                for (proj_name, tensor_type), result in zip(proj_order, results):
                    getattr(cache, tensor_type)[proj_name] = result
            else:
                for proj_name, (new_w, new_s, new_b) in loaded.items():
                    w = cache.weights.pop(proj_name)
                    w[slot_indices] = new_w
                    cache.weights[proj_name] = w

                    s = cache.scales.pop(proj_name)
                    s[slot_indices] = new_s
                    cache.scales[proj_name] = s

                    if cache.biases[proj_name] is not None and new_b is not None:
                        b = cache.biases.pop(proj_name)
                        b[slot_indices] = new_b
                        cache.biases[proj_name] = b
            t_scatter += time.perf_counter() - t_scatter_start

        del shard

    # Rebuild lookup tables and eval scatter results in batches.
    # Scatter graphs were built but not eval'd — eval here triggers the
    # Metal scatter kernels and frees old buffers.
    t_lookup_start = time.perf_counter()
    EVAL_BATCH = 10
    batch_eval = []
    batch_count = 0

    for layer_i in layer_swaps:
        cache = layer_caches[layer_i]
        swaps = layer_swaps[layer_i]

        for slot, old_eid, new_eid in swaps:
            cache.cached_set.discard(old_eid)
            cache.cached_set.add(new_eid)
            cache.cached_ids[slot] = new_eid
            cache.frequency.pop(old_eid, None)
            cache.last_active.pop(old_eid, None)

        lookup_np = np.zeros(cache.num_experts, dtype=np.int32)
        for slot, eid in enumerate(cache.cached_ids):
            lookup_np[eid] = slot
        cache.lookup = mx.array(lookup_np)

        batch_eval.append(cache.lookup)
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            batch_eval.append(cache.weights[proj_name])
            batch_eval.append(cache.scales[proj_name])
            if cache.biases[proj_name] is not None:
                batch_eval.append(cache.biases[proj_name])
        batch_count += 1

        if batch_count >= EVAL_BATCH:
            mx.eval(*batch_eval)
            batch_eval = []
            batch_count = 0

    if batch_eval:
        mx.eval(*batch_eval)

    t_lookup_rebuild = time.perf_counter() - t_lookup_start
    t_rebuild = time.perf_counter() - t1

    # Reset fallback counters
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            proj._cache.total_requests = 0
            proj._cache.total_fallbacks = 0
            proj._cache._indices_buffer.clear()

    mx.clear_cache()

    layers_swapped = len(layer_swaps)
    layers_skipped = sum(1 for s in per_layer_stats
                         if s["missing"] > 0 and s["swapped"] == 0)

    return {
        "discovery_time": t_discovery,
        "rebuild_time": t_rebuild,
        "shard_load_time": t_shard_load,
        "scatter_time": t_scatter,
        "lookup_rebuild_time": t_lookup_rebuild,
        "total_time": t_discovery + t_rebuild,
        "total_swaps": total_swaps,
        "total_missing": total_missing,
        "layers_swapped": layers_swapped,
        "layers_skipped": layers_skipped,
        "discovery_method": discovery_method,
        "per_layer": per_layer_stats,
    }


# ---------------------------------------------------------------------------
# Incremental (async) delta warmup
# ---------------------------------------------------------------------------

@dataclass
class LayerSwapPlan:
    layer_idx: int
    cache: PredictiveExpertCache
    swaps: list  # [(slot, old_eid, new_eid), ...]
    miss_count: int

    def __post_init__(self):
        self.new_eids = mx.array([new_eid for _, _, new_eid in self.swaps])
        self.slot_indices = mx.array([slot for slot, _, _ in self.swaps])


class IncrementalDeltaWarmup:
    """Progressive expert cache updates between tokens.

    After discover(), call step() between generated tokens to incrementally
    swap experts. Each step() builds lazy scatter graphs for N layers —
    no mx.eval(), the forward pass evaluates them naturally.

    Usage:
        warmup = IncrementalDeltaWarmup(model, tokenizer, model_path)
        stats = warmup.discover(new_prompt)

        for response in mlx_lm.stream_generate(model, tokenizer, new_prompt, ...):
            print(response.text, end='', flush=True)
            if not warmup.is_complete:
                warmup.step()
    """

    def __init__(self, model, tokenizer, model_path):
        self._model = model
        self._tokenizer = tokenizer
        self._model_path = Path(model_path)
        self._shard_map = _build_shard_map(self._model_path)
        self._swap_queue: list[LayerSwapPlan] = []
        self._layers_done = 0
        self._swaps_done = 0
        self._total_layers = 0
        self._total_swaps = 0
        self._memory_pressure = False

    def discover(self, prompt, tokens=10):
        """Run discovery pass and compute swap plans.

        Generates tokens through the existing predictive cache to discover
        which experts the new prompt needs, then computes per-layer swap
        plans sorted by miss count (highest first).

        Returns dict with discovery stats.
        """
        import time
        import mlx_lm as _mlx_lm

        # Clear stale indices
        for layer in self._model.layers:
            switch, _ = _find_switch_mlp(layer)
            if switch is None:
                continue
            proj = getattr(switch, "up_proj", None)
            if isinstance(proj, (PredictiveCachedSwitchLinear,
                                 SyncPredictiveCachedSwitchLinear)):
                proj._cache._indices_buffer.clear()

        t0 = time.perf_counter()
        with _with_cache_limit_zero():
            _mlx_lm.generate(self._model, self._tokenizer, prompt=prompt,
                             max_tokens=tokens, verbose=False)

        # Drain indices buffers
        discovered = {}
        for i, layer in enumerate(self._model.layers):
            switch, _ = _find_switch_mlp(layer, i)
            if switch is None:
                continue
            proj = getattr(switch, "up_proj", None)
            if not isinstance(proj, (PredictiveCachedSwitchLinear,
                                     SyncPredictiveCachedSwitchLinear)):
                continue
            cache = proj._cache
            requested = set()
            for indices in cache._indices_buffer:
                flat = np.asarray(indices.reshape(-1))
                requested |= set(int(x) for x in np.unique(flat))
            cache._indices_buffer.clear()
            discovered[i] = requested

        t_discovery = time.perf_counter() - t0

        # Memory pressure check: limit swaps per layer if near device limit.
        # Each swap loads a shard (~336 MB transient), so under pressure we
        # process fewer swaps per step() to avoid pushing past the cliff.
        device_mem = mx.metal.device_info()["memory_size"]
        active_mem = mx.metal.get_active_memory()
        self._memory_pressure = active_mem > 0.85 * device_mem
        if self._memory_pressure:
            print(f"  [memory pressure: {active_mem / 1e9:.1f}/{device_mem / 1e9:.0f} GB — "
                  f"swap plans will be trimmed]")

        # Compute swap plans
        self._swap_queue = []
        total_missing = 0
        max_swaps_per_plan = 3 if self._memory_pressure else 999

        for i, layer in enumerate(self._model.layers):
            switch, _ = _find_switch_mlp(layer, i)
            if switch is None:
                continue
            proj = getattr(switch, "up_proj", None)
            if not isinstance(proj, (PredictiveCachedSwitchLinear,
                                     SyncPredictiveCachedSwitchLinear)):
                continue

            cache = proj._cache
            requested = discovered.get(i, set())
            missing = requested - cache.cached_set
            total_missing += len(missing)

            if not missing:
                continue

            cold = sorted(
                [(cache._lcp_priority(eid), slot, eid)
                 for slot, eid in enumerate(cache.cached_ids)
                 if eid not in requested],
            )

            swaps = []
            for new_eid in sorted(missing):
                if not cold:
                    break
                _, slot, old_eid = cold.pop(0)
                swaps.append((slot, old_eid, new_eid))

            swaps = swaps[:max_swaps_per_plan]

            if swaps:
                self._swap_queue.append(LayerSwapPlan(
                    layer_idx=i, cache=cache,
                    swaps=swaps, miss_count=len(missing),
                ))

        # Sort by miss count descending — fix highest-impact layers first
        self._swap_queue.sort(key=lambda p: p.miss_count, reverse=True)
        self._total_layers = len(self._swap_queue)
        self._total_swaps = sum(p.miss_count for p in self._swap_queue)
        self._layers_done = 0
        self._swaps_done = 0

        # Reset fallback counters
        for layer in self._model.layers:
            switch, _ = _find_switch_mlp(layer)
            if switch is None:
                continue
            proj = getattr(switch, "up_proj", None)
            if isinstance(proj, (PredictiveCachedSwitchLinear,
                                 SyncPredictiveCachedSwitchLinear)):
                proj._cache.total_requests = 0
                proj._cache.total_fallbacks = 0
                proj._cache._indices_buffer.clear()

        return {
            "discovery_time": t_discovery,
            "total_layers": self._total_layers,
            "total_swaps": self._total_swaps,
            "total_missing": total_missing,
        }

    def step(self, layers_per_step=2):
        """Swap experts for the next N layers. All lazy — no mx.eval().

        Constructs scatter graphs that get evaluated naturally by the next
        forward pass. Call between tokens in the generation loop.

        Returns number of layers processed in this step.
        """
        processed = 0
        for _ in range(layers_per_step):
            if not self._swap_queue:
                break

            plan = self._swap_queue.pop(0)
            cache = plan.cache

            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                shard_path = cache._shard_paths[proj_name]
                key_prefix = cache._key_prefixes[proj_name]
                shard = mx.load(shard_path)
                new_w, new_s, new_b = _load_proj_experts(shard, key_prefix, plan.new_eids,
                                                          shard_map=cache._shard_map)
                del shard

                w = cache.weights.pop(proj_name)
                w[plan.slot_indices] = new_w
                cache.weights[proj_name] = w

                s = cache.scales.pop(proj_name)
                s[plan.slot_indices] = new_s
                cache.scales[proj_name] = s

                if cache.biases[proj_name] is not None and new_b is not None:
                    b = cache.biases.pop(proj_name)
                    b[plan.slot_indices] = new_b
                    cache.biases[proj_name] = b

            # Update lookup table
            for slot, old_eid, new_eid in plan.swaps:
                cache.cached_set.discard(old_eid)
                cache.cached_set.add(new_eid)
                cache.cached_ids[slot] = new_eid
                cache.frequency.pop(old_eid, None)
                cache.last_active.pop(old_eid, None)

            lookup_np = np.zeros(cache.num_experts, dtype=np.int32)
            for slot, eid in enumerate(cache.cached_ids):
                lookup_np[eid] = slot
            cache.lookup = mx.array(lookup_np)

            self._layers_done += 1
            self._swaps_done += len(plan.swaps)
            processed += 1

        return processed

    @property
    def is_complete(self):
        return not self._swap_queue

    @property
    def remaining_layers(self):
        return len(self._swap_queue)

    @property
    def total_layers(self):
        return self._total_layers

    @property
    def progress(self):
        return {
            "layers_done": self._layers_done,
            "layers_total": self._total_layers,
            "swaps_done": self._swaps_done,
            "swaps_total": self._total_swaps,
        }


def incremental_delta_warmup(model, tokenizer, model_path, new_prompt,
                              discovery_tokens=10):
    """Create an IncrementalDeltaWarmup and run discovery.

    Returns the warmup object ready for step() calls between tokens.
    """
    warmup = IncrementalDeltaWarmup(model, tokenizer, model_path)
    stats = warmup.discover(new_prompt, tokens=discovery_tokens)
    return warmup, stats


# ---------------------------------------------------------------------------
# Adaptive per-layer capacity
# ---------------------------------------------------------------------------

def adaptive_capacity_upgrade(model, model_path, total_budget_experts,
                              min_per_layer=32, sync=False):
    """Compute per-layer capacities from LCP warmup and upgrade to predictive.

    After LCP warmup, inspects each layer's discovered expert count and allocates
    capacity proportionally (with 30% headroom), subject to total budget and min floor.
    Layers that discover more experts get more capacity.

    Returns dict with allocations, discovered counts, and memory estimate.
    """
    layer_counts = []
    moe_layers = []

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "gate_proj")
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue
        layer_counts.append(len(proj._cache.all_seen))
        moe_layers.append(i)

    n = len(layer_counts)

    # Raw = discovered + 30% headroom, at least min_per_layer
    raw = [max(min_per_layer, int(count * 1.3)) for count in layer_counts]
    total_raw = sum(raw)

    # Scale proportionally to fit total budget
    scale = total_budget_experts / total_raw if total_raw > 0 else 1.0
    capacities = [max(min_per_layer, min(512, round(r * scale))) for r in raw]

    # Fine-tune to hit exact budget
    diff = total_budget_experts - sum(capacities)
    sorted_idx = sorted(range(n), key=lambda j: capacities[j], reverse=True)
    for j in sorted_idx:
        if diff == 0:
            break
        if diff > 0:
            if capacities[j] < 512:
                capacities[j] += 1
                diff -= 1
        elif capacities[j] > min_per_layer:
            capacities[j] -= 1
            diff += 1

    upgraded = upgrade_to_predictive(model, model_path, capacities, sync=sync)

    total_experts = sum(capacities)
    estimated_gb = sum(c * 1.769 for c in capacities) / 1000

    return {
        "upgraded": upgraded,
        "allocations": dict(zip(moe_layers, capacities)),
        "discovered": dict(zip(moe_layers, layer_counts)),
        "capacities": capacities,
        "total_experts": total_experts,
        "estimated_memory_gb": estimated_gb,
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def measure_fallback(model) -> dict:
    """Compute fallback stats by draining buffered indices post-generation.

    Unlike get_fallback_stats() (which reads counters updated by cache.update()),
    this directly inspects which requested expert IDs are not in the cached set.
    Call after mlx_lm.generate() completes.
    """
    total_requests = 0
    total_fallbacks = 0
    layer_stats = []

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue

        cache = proj._cache
        all_requested = set()
        for indices in cache._indices_buffer:
            flat = np.asarray(indices.reshape(-1))
            all_requested |= set(int(x) for x in np.unique(flat))
        cache._indices_buffer.clear()

        missing = all_requested - cache.cached_set
        n_req = len(all_requested)
        n_fb = len(missing)
        total_requests += n_req
        total_fallbacks += n_fb

        if n_req > 0:
            layer_stats.append({
                "layer": i,
                "requested": n_req,
                "missing": n_fb,
                "fallback_rate": n_fb / n_req,
            })

    return {
        "total_requests": total_requests,
        "total_fallbacks": total_fallbacks,
        "fallback_rate": total_fallbacks / total_requests if total_requests > 0 else 0.0,
        "layers": layer_stats,
    }


def get_cache_stats(model) -> dict:
    """Collect hit/miss stats from all ExpertCache instances in the model."""
    total_hits = 0
    total_misses = 0
    layer_stats = []

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue
        cache = proj._cache
        hits = cache.hits
        misses = cache.misses
        total = hits + misses
        rate = hits / total if total > 0 else 0.0
        layer_stats.append({
            "layer": i,
            "hits": hits,
            "misses": misses,
            "hit_rate": rate,
            "cached_experts": len(cache.entries),
        })
        total_hits += hits
        total_misses += misses

    total = total_hits + total_misses
    return {
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_hit_rate": total_hits / total if total > 0 else 0.0,
        "layers": layer_stats,
    }


# ---------------------------------------------------------------------------
# Router-only discovery (fast warmup alternatives)
# ---------------------------------------------------------------------------

def router_only_forward(model, tokenizer, prompt, max_tokens=10):
    """Run the model with MoE expert computation skipped, collecting router selections.

    Monkey-patches MoE blocks to run the gate (router) and shared expert but
    skip switch_mlp. Hidden states drift without MoE output, but routers still
    produce plausible expert selections. Works with any model architecture that
    uses _find_moe_block-detectable MoE layers.

    Returns dict[layer_idx, set[expert_id]] of all experts selected across all tokens.
    """
    import mlx_lm as _mlx_lm

    collected: dict[int, set[int]] = {}
    moe_blocks: dict[int, int] = {}  # id(block) -> layer_idx

    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and hasattr(block, "switch_mlp"):
            moe_blocks[id(block)] = i
            collected[i] = set()

    if not moe_blocks:
        return collected

    # Group by type for monkey-patching
    type_to_blocks: dict[type, list] = {}
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and id(block) in moe_blocks:
            type_to_blocks.setdefault(type(block), []).append(block)

    original_calls: dict[type, object] = {}

    def _make_skip_call(block_map, orig_call):
        def _skip(self, x):
            layer_idx = block_map.get(id(self))
            if layer_idx is None:
                return orig_call(self, x)

            gates = self.gate(x)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            mx.eval(inds)
            flat = np.asarray(inds.reshape(-1))
            collected[layer_idx].update(int(e) for e in flat)

            if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
                shared_y = self.shared_expert(x)
                shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
                return shared_y
            elif hasattr(self, "shared_expert"):
                return self.shared_expert(x)
            else:
                return mx.zeros_like(x)
        return _skip

    for block_type, blocks in type_to_blocks.items():
        original_calls[block_type] = block_type.__call__
        block_type.__call__ = _make_skip_call(moe_blocks, original_calls[block_type])

    try:
        _mlx_lm.generate(model, tokenizer, prompt=prompt,
                         max_tokens=max_tokens, verbose=False)
    finally:
        for block_type, orig in original_calls.items():
            block_type.__call__ = orig

    return collected


def router_only_discovery(model, tokenizer, prompt, max_tokens=10):
    """Fast cold-start discovery: run routers only, populate Phase 2 caches.

    Like router_only_forward() but batches all mx.eval to the end instead of
    sync-ing per layer per token (480 sync points -> 1). Populates the
    CachedQuantizedSwitchLinear caches so upgrade_to_predictive() can use them.

    Returns dict[layer_idx, set[expert_id]].
    """
    import mlx_lm as _mlx_lm

    collected: dict[int, list[mx.array]] = {}
    moe_blocks: dict[int, int] = {}

    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and hasattr(block, "switch_mlp"):
            moe_blocks[id(block)] = i
            collected[i] = []

    if not moe_blocks:
        return {}

    type_to_blocks: dict[type, list] = {}
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and id(block) in moe_blocks:
            type_to_blocks.setdefault(type(block), []).append(block)

    original_calls: dict[type, object] = {}

    def _make_skip_call(block_map, orig_call):
        def _skip(self, x):
            layer_idx = block_map.get(id(self))
            if layer_idx is None:
                return orig_call(self, x)

            gates = self.gate(x)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            collected[layer_idx].append(inds)

            if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
                shared_y = self.shared_expert(x)
                shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
                return shared_y
            elif hasattr(self, "shared_expert"):
                return self.shared_expert(x)
            else:
                return mx.zeros_like(x)
        return _skip

    for block_type, blocks in type_to_blocks.items():
        original_calls[block_type] = block_type.__call__
        block_type.__call__ = _make_skip_call(moe_blocks, original_calls[block_type])

    try:
        _mlx_lm.generate(model, tokenizer, prompt=prompt,
                         max_tokens=max_tokens, verbose=False)
    finally:
        for block_type, orig in original_calls.items():
            block_type.__call__ = orig

    # One bulk eval for all collected indices
    all_tensors = []
    for inds_list in collected.values():
        all_tensors.extend(inds_list)
    if all_tensors:
        mx.eval(*all_tensors)

    # Flatten into sets and populate Phase 2 caches
    result: dict[int, set[int]] = {}
    for i, inds_list in collected.items():
        expert_counts: dict[int, int] = {}
        for inds in inds_list:
            flat = np.asarray(inds.reshape(-1))
            for eid in flat:
                eid = int(eid)
                expert_counts[eid] = expert_counts.get(eid, 0) + 1
        result[i] = set(expert_counts.keys())

        # Populate Phase 2 LCP cache
        layer = model.layers[i]
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "gate_proj", None)
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue
        cache = proj._cache
        for eid, count in expert_counts.items():
            cache.entries[eid] = {}
            cache.frequency[eid] = count
            cache.last_active[eid] = max_tokens
            cache.all_seen.add(eid)
        cache.step = max_tokens

    return result


def speculative_router_probe(model, tokenizer, prompt, max_tokens=10):
    """Skip-MoE forward, then probe each router on ALL layers' hidden states.

    Runs a skip-MoE forward pass (same as router_only_forward) to capture
    per-layer hidden states cheaply. Then probes each MoE layer's router on
    the UNION of all layers' captured hidden states, not just its own.

    This tests whether hidden states from other layers help discover experts
    that the layer's own drifted state misses.

    Returns dict[layer_idx, set[expert_id]].
    """
    import mlx_lm as _mlx_lm

    moe_layer_indices: list[int] = []
    moe_blocks_map: dict[int, int] = {}  # id(block) -> layer_idx
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and hasattr(block, "switch_mlp"):
            moe_layer_indices.append(i)
            moe_blocks_map[id(block)] = i

    hidden_states_per_layer: dict[int, list[mx.array]] = {i: [] for i in moe_layer_indices}

    # Group by type for monkey-patching
    type_to_blocks: dict[type, list] = {}
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and id(block) in moe_blocks_map:
            type_to_blocks.setdefault(type(block), []).append(block)

    original_calls: dict[type, object] = {}

    def _make_skip_and_capture(block_map, orig_call):
        def _skip(self, x):
            layer_idx = block_map.get(id(self))
            if layer_idx is None:
                return orig_call(self, x)

            mx.eval(x)
            hidden_states_per_layer[layer_idx].append(x)

            gates = self.gate(x)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]

            if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
                shared_y = self.shared_expert(x)
                shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
                return shared_y
            elif hasattr(self, "shared_expert"):
                return self.shared_expert(x)
            else:
                return mx.zeros_like(x)
        return _skip

    for block_type, blocks in type_to_blocks.items():
        original_calls[block_type] = block_type.__call__
        block_type.__call__ = _make_skip_and_capture(moe_blocks_map, original_calls[block_type])

    try:
        _mlx_lm.generate(model, tokenizer, prompt=prompt,
                         max_tokens=max_tokens, verbose=False)
    finally:
        for block_type, orig in original_calls.items():
            block_type.__call__ = orig

    # Collect ALL hidden states from ALL layers into one pool
    all_states: list[mx.array] = []
    for layer_idx in moe_layer_indices:
        all_states.extend(hidden_states_per_layer[layer_idx])

    # Probe each router on the full pool of hidden states
    collected: dict[int, set[int]] = {}
    for layer_idx in moe_layer_indices:
        moe_block = _find_moe_block(model.layers[layer_idx])
        all_experts = set()
        for h in all_states:
            gates = moe_block.gate(h)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = moe_block.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            mx.eval(inds)
            flat = np.asarray(inds.reshape(-1))
            all_experts.update(int(e) for e in flat)
        collected[layer_idx] = all_experts

    return collected


def speculative_router_cross_layer(model, tokenizer, prompt, max_tokens=10):
    """Probe all routers using only the FIRST MoE layer's hidden states.

    Uses the skip-MoE forward pass. Captures hidden states only at the first
    MoE layer, then feeds those same states through every other MoE layer's
    router. This tests the strongest form of the MoEpic hypothesis: that a
    single layer's hidden states predict all other layers' routing.

    Returns dict[layer_idx, set[expert_id]].
    """
    import mlx_lm as _mlx_lm

    moe_layer_indices: list[int] = []
    moe_blocks_map: dict[int, int] = {}  # id(block) -> layer_idx
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and hasattr(block, "switch_mlp"):
            moe_layer_indices.append(i)
            moe_blocks_map[id(block)] = i

    first_moe = moe_layer_indices[0]
    captured_states: list[mx.array] = []

    # Group by type for monkey-patching
    type_to_blocks: dict[type, list] = {}
    for i, layer in enumerate(model.layers):
        block = _find_moe_block(layer)
        if block is not None and id(block) in moe_blocks_map:
            type_to_blocks.setdefault(type(block), []).append(block)

    original_calls: dict[type, object] = {}

    def _make_skip_and_capture_first(block_map, first_idx, orig_call):
        def _skip(self, x):
            layer_idx = block_map.get(id(self))
            if layer_idx is None:
                return orig_call(self, x)
            if layer_idx == first_idx:
                mx.eval(x)
                captured_states.append(x)

            if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
                shared_y = self.shared_expert(x)
                shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
                return shared_y
            elif hasattr(self, "shared_expert"):
                return self.shared_expert(x)
            else:
                return mx.zeros_like(x)
        return _skip

    for block_type, blocks in type_to_blocks.items():
        original_calls[block_type] = block_type.__call__
        block_type.__call__ = _make_skip_and_capture_first(
            moe_blocks_map, first_moe, original_calls[block_type])

    try:
        _mlx_lm.generate(model, tokenizer, prompt=prompt,
                         max_tokens=max_tokens, verbose=False)
    finally:
        for block_type, orig in original_calls.items():
            block_type.__call__ = orig

    # Probe every router with first layer's hidden states
    collected: dict[int, set[int]] = {}
    for layer_idx in moe_layer_indices:
        moe_block = _find_moe_block(model.layers[layer_idx])
        all_experts = set()
        for h in captured_states:
            gates = moe_block.gate(h)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = moe_block.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            mx.eval(inds)
            flat = np.asarray(inds.reshape(-1))
            all_experts.update(int(e) for e in flat)
        collected[layer_idx] = all_experts

    return collected


def select_capacity(target_model_memory_gb: float, system_memory_gb: float,
                    num_moe_layers: int = 48, expert_slot_mb: float = 1.69) -> int:
    """Select expert cache capacity to stay under the Metal memory pressure cliff."""
    budget_gb = system_memory_gb * 0.56 - target_model_memory_gb
    expert_memory_per_slot_gb = num_moe_layers * expert_slot_mb / 1024
    capacity = int(budget_gb / expert_memory_per_slot_gb)
    capacity = (capacity // 8) * 8
    return max(0, min(512, capacity))


# ---------------------------------------------------------------------------
# Metal cache limit helper
# ---------------------------------------------------------------------------

def _with_cache_limit_zero():
    """Context manager to temporarily set Metal cache limit to 0.

    Reclaims several GB of MLX buffer cache headroom, giving 1.3-2x speedup
    for operations above the 20 GB Metal pressure cliff.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        # MLX has no get_cache_limit(); restore to 25% of device memory (MLX default)
        default_limit = mx.metal.device_info()["memory_size"] // 4
        mx.metal.set_cache_limit(0)
        mx.metal.clear_cache()
        try:
            yield
        finally:
            mx.metal.set_cache_limit(default_limit)

    return _ctx()


# ---------------------------------------------------------------------------
# Cache state persistence (skip warmup on repeated launches)
# ---------------------------------------------------------------------------

def save_cache_state(model, path, metadata=None):
    """Save discovered expert routing state to JSON for fast cold start.

    Captures per-layer expert IDs, frequencies, and LCP priorities from
    the current cache state (Phase 2 ExpertCache or Phase 3 PredictiveExpertCache).
    """
    import datetime

    layers = {}
    capacity = None

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue

        proj = getattr(switch, "gate_proj", None)
        if proj is None:
            continue

        if isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            cache = proj._cache
            layers[str(i)] = {
                "cached_ids": list(cache.cached_ids),
                "frequency": {str(k): v for k, v in cache.frequency.items()},
                "last_active": {str(k): v for k, v in cache.last_active.items()},
                "step": cache.step,
                "all_seen": list(cache.cached_set),
            }
            if capacity is None:
                capacity = cache.capacity

        elif isinstance(proj, CachedQuantizedSwitchLinear):
            cache = proj._cache
            layers[str(i)] = {
                "cached_ids": sorted(cache.entries.keys()),
                "frequency": {str(k): v for k, v in cache.frequency.items()},
                "last_active": {str(k): v for k, v in cache.last_active.items()},
                "step": cache.step,
                "all_seen": sorted(cache.all_seen),
            }
            if capacity is None:
                capacity = cache.capacity

    state = {
        "version": 1,
        "capacity": capacity,
        "timestamp": datetime.datetime.now().isoformat(),
        "metadata": metadata or {},
        "layers": layers,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f)


def load_cache_state(path):
    """Load saved expert routing state from JSON."""
    with open(path) as f:
        state = json.load(f)
    if state.get("version") != 1:
        raise ValueError(f"Unsupported cache state version: {state.get('version')}")
    return state


def save_prepacked_weights(model, path):
    """Save pre-stacked predictive cache weights to safetensors for fast warm start.

    After upgrade_to_predictive(), the PredictiveExpertCache objects have stacked
    weight/scale/bias tensors in Metal memory. Save them to disk so the next warm
    start can load_prepacked_weights() and skip the entire upgrade_to_predictive().

    Convention: cache at "foo.json" -> weights at "foo.weights.safetensors",
    metadata at "foo.weights.meta.json".
    """
    path = Path(path)
    tensors: dict[str, mx.array] = {}
    meta_layers: dict[str, dict] = {}

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "gate_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue

        cache = proj._cache
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"layer.{i}.{proj_name}.weight"] = cache.weights[proj_name]
            tensors[f"layer.{i}.{proj_name}.scales"] = cache.scales[proj_name]
            if cache.biases[proj_name] is not None:
                tensors[f"layer.{i}.{proj_name}.biases"] = cache.biases[proj_name]

        meta_layers[str(i)] = {
            "cached_ids": list(cache.cached_ids),
            "num_experts": cache.num_experts,
            "pinned_set": sorted(cache.pinned_set),
            "frequency": {str(k): v for k, v in cache.frequency.items()},
            "last_active": {str(k): v for k, v in cache.last_active.items()},
            "step": cache.step,
        }

    mx.save_safetensors(str(path), tensors)

    meta_path = Path(str(path) + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump({"version": 1, "layers": meta_layers}, f)

    print(f"  Saved prepacked weights: {path} ({len(tensors)} tensors, "
          f"{len(meta_layers)} layers)")


def load_prepacked_weights(model, prepacked_path, model_path=None):
    """Load pre-stacked predictive cache from safetensors, skipping upgrade_to_predictive().

    The model must already have Phase 2 modules installed (enable_lazy_experts with
    predictive=True). This replaces them with PredictiveCachedSwitchLinear using the
    pre-packed tensors directly.

    Returns number of modules upgraded.
    """
    prepacked_path = Path(prepacked_path)
    meta_path = Path(str(prepacked_path) + ".meta.json")

    with open(meta_path) as f:
        meta = json.load(f)

    packed = mx.load(str(prepacked_path))

    shard_map = None
    if model_path is not None:
        shard_map = _build_shard_map(Path(model_path))

    upgraded = 0
    layers_processed = 0
    batch_eval: list[mx.array] = []

    for layer_str, layer_meta in meta["layers"].items():
        i = int(layer_str)
        layer = model.layers[i]
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is None:
            continue

        cached_ids = layer_meta["cached_ids"]
        num_experts = layer_meta["num_experts"]
        capacity = len(cached_ids)

        pred_cache = PredictiveExpertCache(capacity, num_experts)

        # Grab group_size/bits/mode from current Phase 2 module
        phase2_mod = getattr(switch, "gate_proj")

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            w_key = f"layer.{i}.{proj_name}.weight"
            s_key = f"layer.{i}.{proj_name}.scales"
            b_key = f"layer.{i}.{proj_name}.biases"

            pred_cache.weights[proj_name] = packed[w_key]
            pred_cache.scales[proj_name] = packed[s_key]
            pred_cache.biases[proj_name] = packed[b_key] if b_key in packed else None

            batch_eval.append(pred_cache.weights[proj_name])
            batch_eval.append(pred_cache.scales[proj_name])
            if pred_cache.biases[proj_name] is not None:
                batch_eval.append(pred_cache.biases[proj_name])

        pred_cache.build_lookup(cached_ids)
        batch_eval.append(pred_cache.lookup)

        pred_cache.pinned_set = set(layer_meta.get("pinned_set", []))
        pred_cache.frequency = {int(k): v for k, v in layer_meta.get("frequency", {}).items()}
        pred_cache.last_active = {int(k): v for k, v in layer_meta.get("last_active", {}).items()}
        pred_cache.step = layer_meta.get("step", 0)

        if shard_map is not None and key_base is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                key_prefix = f"{key_base}.{proj_name}"
                pred_cache._shard_paths[proj_name] = shard_map[f"{key_prefix}.weight"]
                pred_cache._key_prefixes[proj_name] = key_prefix
            pred_cache._shard_map = shard_map

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            p2 = getattr(switch, proj_name)
            replacement = PredictiveCachedSwitchLinear(
                group_size=p2.group_size,
                bits=p2.bits,
                mode=p2.mode,
                proj_name=proj_name,
                cache=pred_cache,
            )
            setattr(switch, proj_name, replacement)
            upgraded += 1

        layers_processed += 1

        # Eval in chunks of 8 layers to bound transient memory
        if layers_processed % 8 == 0:
            mx.eval(*batch_eval)
            batch_eval = []
            print(f"  Prepacked load: {layers_processed} layers "
                  f"({mx.get_active_memory() / 1e9:.1f} GB)")

    if batch_eval:
        mx.eval(*batch_eval)

    del packed

    print(f"  Prepacked load complete: {layers_processed} layers, {upgraded} modules "
          f"({mx.get_active_memory() / 1e9:.1f} GB)")
    return upgraded


def upgrade_from_saved_state(model, model_path, cache_state, capacity, sync=False):
    """Skip warmup: build predictive cache directly from saved state.

    Instead of: enable_lazy → warmup gen → upgrade_to_predictive
    Does: enable_lazy → load saved state → upgrade_to_predictive

    The model must already have Phase 2 modules installed via
    enable_lazy_experts(predictive=True). This populates the Phase 2 caches
    with frequency/last_active/all_seen from the saved state, then calls
    upgrade_to_predictive() which loads weights from disk.

    Returns number of modules upgraded.
    """
    # Memory guard: check if projected expert memory fits in 85% of device RAM
    num_moe_layers = sum(
        1 for layer in model.layers
        if _find_switch_mlp(layer)[0] is not None
    )
    expert_slot_mb = 1.69
    base_memory_gb = mx.metal.get_active_memory() / 1e9
    projected_gb = base_memory_gb + capacity * num_moe_layers * expert_slot_mb / 1024
    device_gb = mx.metal.device_info()["memory_size"] / 1e9
    limit_gb = 0.85 * device_gb

    if projected_gb > limit_gb:
        max_capacity = int((limit_gb - base_memory_gb) * 1024 / (num_moe_layers * expert_slot_mb))
        max_capacity = (max_capacity // 8) * 8
        max_capacity = max(max_capacity, 0)
        print(f"  [memory guard: {projected_gb:.1f} GB projected > {limit_gb:.1f} GB limit — "
              f"reducing capacity {capacity} -> {max_capacity}]")
        capacity = max_capacity

    layers_data = cache_state["layers"]

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue

        proj = getattr(switch, "gate_proj", None)
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue

        layer_key = str(i)
        if layer_key not in layers_data:
            continue

        saved = layers_data[layer_key]
        cache = proj._cache

        cache.frequency = {int(k): v for k, v in saved["frequency"].items()}
        cache.last_active = {int(k): v for k, v in saved["last_active"].items()}
        cache.step = saved["step"]
        cache.all_seen = set(saved["all_seen"])

        # Put empty entries so upgrade_to_predictive sees the right expert keys
        # for ranking. lookup() returns None -> all weights loaded from disk.
        for eid in saved["all_seen"]:
            cache.entries[eid] = {}

    return upgrade_to_predictive(model, model_path, capacity, sync=sync)

# ---------------------------------------------------------------------------
# Universal expert profiling + pinned cache partition (Task 3)
# ---------------------------------------------------------------------------

def load_universal_profile(path):
    """Load universal expert profile from JSON."""
    with open(path) as f:
        return json.load(f)


def upgrade_to_predictive_with_pinning(model, model_path, capacity,
                                        universal_profile, pin_threshold=0.5,
                                        sync=False):
    """Like upgrade_to_predictive but pins universal experts.

    Universal experts occupy the first N slots and are marked as non-evictable.
    Remaining slots are filled with LCP-ranked discovered experts (evictable).

    Args:
        universal_profile: Dict from load_universal_profile() or profile_experts.py.
        pin_threshold: Minimum activation fraction to consider an expert universal.
        sync: If True, use SyncPredictiveCachedSwitchLinear.

    Returns number of modules upgraded.
    """
    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)
    num_prompts = universal_profile["num_prompts"]
    min_count = int(pin_threshold * num_prompts)

    # Build per-layer universal expert lists from the profile
    universal_per_layer: dict[int, list[int]] = {}
    for layer_str, layer_data in universal_profile["layers"].items():
        layer_idx = int(layer_str)
        counts = layer_data.get("activation_counts", {})
        universal = sorted(
            int(eid) for eid, cnt in counts.items()
            if int(cnt) >= min_count
        )
        universal_per_layer[layer_idx] = universal

    # Pass 1: harvest LCP caches with pinned experts in front
    layer_meta = {}
    for i, layer in enumerate(model.layers):
        switch, key_base = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        first_proj = getattr(switch, "gate_proj")
        if not isinstance(first_proj, CachedQuantizedSwitchLinear):
            continue

        lcp_cache = first_proj._cache
        num_experts = _detect_num_experts(switch)
        C = min(capacity, num_experts)

        pinned = universal_per_layer.get(i, [])[:C]
        pinned_set_local = set(pinned)
        n_pinned = len(pinned)

        # Remaining slots: LCP-ranked discovered experts (excluding pinned)
        discovered = sorted(
            (eid for eid in lcp_cache.entries.keys() if eid not in pinned_set_local),
            key=lambda eid: lcp_cache._priority(eid),
            reverse=True,
        )[:C - n_pinned]
        discovered_set = set(discovered) | pinned_set_local

        filler = []
        for eid in range(num_experts):
            if len(pinned) + len(discovered) + len(filler) >= C:
                break
            if eid not in discovered_set:
                filler.append(eid)

        cached_ids = list(pinned) + list(discovered) + filler

        pred_cache = PredictiveExpertCache(C, num_experts)
        harvested = {}
        to_load = {}
        has_bias = None
        phase2_mods = {}

        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mods[name] = getattr(switch, name)
            h_list = []
            load_list = []
            for slot, eid in enumerate(cached_ids):
                cached = lcp_cache.lookup(eid, name)
                if cached is not None:
                    w, s, b = cached
                    if has_bias is None:
                        has_bias = b is not None
                    h_list.append((slot, w, s, b))
                else:
                    load_list.append((slot, eid))
            harvested[name] = h_list
            to_load[name] = load_list

        layer_meta[i] = {
            "cached_ids": cached_ids,
            "pred_cache": pred_cache,
            "harvested": harvested,
            "to_load": to_load,
            "has_bias": has_bias if has_bias is not None else True,
            "phase2_mods": phase2_mods,
            "lcp_cache": lcp_cache,
            "n_pinned": n_pinned,
            "pinned_set": pinned_set_local,
            "C": C,
            "key_base": key_base,
        }

    # Pass 2: group disk loads by shard, load each shard once
    shard_groups: dict[str, list[tuple]] = {}
    for i, meta in layer_meta.items():
        for name in ("gate_proj", "up_proj", "down_proj"):
            if not meta["to_load"][name]:
                continue
            key_prefix = f"{meta['key_base']}.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            shard_groups.setdefault(shard_path, []).append(
                (i, name, key_prefix, meta["to_load"][name]))

    loaded: dict[int, dict[str, dict[int, tuple]]] = {}

    for shard_path, group in shard_groups.items():
        shard = mx.load(shard_path)
        layers_in_batch = sorted(set(layer_i for layer_i, _, _, _ in group))
        for layer_i in layers_in_batch:
            layer_entries = [(n, kp, slots) for li, n, kp, slots in group if li == layer_i]
            to_eval = []
            for name, key_prefix, slot_eids in layer_entries:
                load_ids = mx.array([eid for _, eid in slot_eids])
                w_batch, s_batch, b_batch = _load_proj_experts(shard, key_prefix, load_ids,
                                                              shard_map=shard_map)
                to_eval.extend([w_batch, s_batch])
                if b_batch is not None:
                    to_eval.append(b_batch)

                slot_map = {}
                for j, (slot, _) in enumerate(slot_eids):
                    slot_map[slot] = (w_batch[j], s_batch[j],
                                      b_batch[j] if b_batch is not None else None)
                loaded.setdefault(layer_i, {})[name] = slot_map

            mx.eval(*to_eval)
        del shard

    # Pass 3: assemble stacked tensors, build lookups, install modules
    upgraded = 0
    cls = SyncPredictiveCachedSwitchLinear if sync else PredictiveCachedSwitchLinear

    for i, meta in layer_meta.items():
        pred_cache = meta["pred_cache"]
        cached_ids = meta["cached_ids"]
        has_bias = meta["has_bias"]
        C = meta["C"]

        for name in ("gate_proj", "up_proj", "down_proj"):
            ws, ss, bs = [], [], []
            harvested_map = {slot: (w, s, b) for slot, w, s, b in meta["harvested"][name]}
            loaded_map = loaded.get(i, {}).get(name, {})

            for slot in range(C):
                if slot in harvested_map:
                    w, s, b = harvested_map[slot]
                else:
                    w, s, b = loaded_map[slot]
                ws.append(w)
                ss.append(s)
                if has_bias:
                    bs.append(b)

            pred_cache.weights[name] = mx.stack(ws)
            pred_cache.scales[name] = mx.stack(ss)
            pred_cache.biases[name] = mx.stack(bs) if has_bias else None

        key_base = meta["key_base"]
        for name in ("gate_proj", "up_proj", "down_proj"):
            key_prefix = f"{key_base}.{name}"
            pred_cache._shard_paths[name] = shard_map[f"{key_prefix}.weight"]
            pred_cache._key_prefixes[name] = key_prefix
        pred_cache._shard_map = shard_map

        pred_cache.build_lookup(cached_ids)
        pred_cache.pinned_set = meta["pinned_set"]
        mx.eval(pred_cache.lookup)

        switch, _ = _find_switch_mlp(model.layers[i], i)
        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mod = meta["phase2_mods"][name]
            replacement = cls(
                group_size=phase2_mod.group_size,
                bits=phase2_mod.bits,
                mode=phase2_mod.mode,
                proj_name=name,
                cache=pred_cache,
            )
            setattr(switch, name, replacement)
            upgraded += 1

        meta["lcp_cache"].entries.clear()
        meta["lcp_cache"].frequency.clear()
        meta["lcp_cache"].last_active.clear()

        print(f"  Layer {i}: {meta['n_pinned']} pinned + "
              f"{C - meta['n_pinned']} dynamic = {C} experts "
              f"({mx.get_active_memory() / 1e9:.1f} GB)")

    return upgraded


def upgrade_from_profile(model, model_path, capacity, profile, pin_threshold=0.5):
    """Profile-based cold start: skip discovery, use profile's top experts directly.

    When a universal expert profile exists, populates Phase 2 caches from the
    profile's activation counts and calls upgrade_to_predictive(). Experts above
    pin_threshold are marked pinned after upgrade.

    The model must already have Phase 2 modules installed (enable_lazy_experts
    with predictive=True).

    Returns number of modules upgraded.
    """
    model_path = Path(model_path)
    num_prompts = profile["num_prompts"]
    min_count = int(pin_threshold * num_prompts)

    moe_idx = 0
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "gate_proj", None)
        if not isinstance(proj, CachedQuantizedSwitchLinear):
            continue

        layer_data = profile["layers"].get(str(i))
        if layer_data is None:
            moe_idx += 1
            continue

        counts = layer_data.get("activation_counts", {})
        sorted_experts = sorted(
            ((int(eid), int(cnt)) for eid, cnt in counts.items()),
            key=lambda x: x[1],
            reverse=True,
        )

        cache = proj._cache
        for eid, count in sorted_experts:
            cache.entries[eid] = {}
            cache.frequency[eid] = count
            cache.last_active[eid] = 1
            cache.all_seen.add(eid)
        cache.step = 1
        moe_idx += 1

    upgraded = upgrade_to_predictive(model, model_path, capacity)

    # Set pinned_set on the newly-installed PredictiveExpertCaches
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "gate_proj", None)
        if not isinstance(proj, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue

        layer_data = profile["layers"].get(str(i))
        if layer_data is None:
            continue

        counts = layer_data.get("activation_counts", {})
        pinned = set(
            int(eid) for eid, cnt in counts.items()
            if int(cnt) >= min_count
        )
        proj._cache.pinned_set = pinned & proj._cache.cached_set

    return upgraded


# ---------------------------------------------------------------------------
# Per-layer adaptive cache budget via MoEpic greedy (Task 6)
# ---------------------------------------------------------------------------

def compute_adaptive_allocations(layer_profiles, total_budget, min_per_layer=32):
    """Compute optimal per-layer expert cache allocations using MoEpic greedy.

    Iteratively transfers one slot from the layer with lowest marginal cost
    to the layer with highest marginal utility.

    Args:
        layer_profiles: Dict mapping layer_idx to profile dict with:
            - "working_set": list of (expert_id, activation_count) sorted descending
            - "entropy": float (routing entropy)
            - "unique_count": int
        total_budget: Total expert slots to allocate across all layers.
        min_per_layer: Minimum slots per layer.

    Returns dict with allocations, miss_rates, and iterations.
    """
    layers = sorted(layer_profiles.keys())
    n_layers = len(layers)

    base = max(min_per_layer, total_budget // n_layers)
    allocs = {li: min(base, 512) for li in layers}

    current_total = sum(allocs.values())
    if current_total < total_budget:
        deficit = total_budget - current_total
        for li in layers:
            if deficit <= 0:
                break
            add = min(deficit, 512 - allocs[li])
            allocs[li] += add
            deficit -= add
    elif current_total > total_budget:
        surplus = current_total - total_budget
        for li in reversed(layers):
            if surplus <= 0:
                break
            remove = min(surplus, allocs[li] - min_per_layer)
            allocs[li] -= remove
            surplus -= remove

    def _miss_rate(layer_idx, cap):
        ws = layer_profiles[layer_idx]["working_set"]
        if not ws or cap >= len(ws):
            return 0.0
        total_activations = sum(cnt for _, cnt in ws)
        if total_activations == 0:
            return 0.0
        covered = sum(cnt for _, cnt in ws[:cap])
        return 1.0 - covered / total_activations

    def _marginal_cost(layer_idx):
        cap = allocs[layer_idx]
        if cap <= min_per_layer:
            return float('inf')
        return _miss_rate(layer_idx, cap - 1) - _miss_rate(layer_idx, cap)

    def _marginal_utility(layer_idx):
        cap = allocs[layer_idx]
        if cap >= 512:
            return 0.0
        return _miss_rate(layer_idx, cap) - _miss_rate(layer_idx, cap + 1)

    max_iterations = total_budget * 2
    iterations = 0
    for _ in range(max_iterations):
        donor = min(layers, key=_marginal_cost)
        recipient = max(layers, key=_marginal_utility)

        if donor == recipient:
            break
        cost = _marginal_cost(donor)
        utility = _marginal_utility(recipient)
        if utility <= cost or utility <= 1e-9:
            break

        allocs[donor] -= 1
        allocs[recipient] += 1
        iterations += 1

    miss_rates = {li: _miss_rate(li, allocs[li]) for li in layers}

    return {
        "allocations": allocs,
        "miss_rates": miss_rates,
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Production one-call API
# ---------------------------------------------------------------------------

def flash_generate(model_name, prompt, max_tokens=200, cache_dir=None,
                   profile_path=None, prepacked=True):
    """One-call generation with all optimizations.

    Auto-detects RAM, selects capacity, loads cached state if available,
    applies pinning if profile exists, uses cache_limit(0) during warmup,
    coherent stream mode for delta switches.

    Warm start priority:
      1. Prepacked weights (fastest: skip upgrade_to_predictive entirely)
      2. Saved cache state (reload expert weights from safetensors shards)

    Cold start priority:
      1. Profile-based (skip discovery, use profile's top experts)
      2. Router-only discovery (fast ~1-2s vs ~75s full model discovery)

    Args:
        model_name: HuggingFace model name (e.g. "mlx-community/Qwen3-Coder-Next-4bit").
        prompt: Text prompt for generation.
        max_tokens: Maximum tokens to generate.
        cache_dir: Directory for cache state persistence. None disables caching.
        profile_path: Path to universal expert profile JSON for pinning.
        prepacked: Save/load prepacked weight files for fastest warm start.

    Returns:
        Generated text string.
    """
    import os
    import time
    import mlx_lm as _mlx_lm
    from mlx_lm.utils import hf_repo_to_path

    t_total_start = time.perf_counter()

    model_path = hf_repo_to_path(model_name)
    t0 = time.perf_counter()
    model, tokenizer = _mlx_lm.load(model_name, lazy=True)

    num_moe_layers = 0
    num_experts = 512
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is not None:
            num_moe_layers += 1
            num_experts = _detect_num_experts(switch)

    device_gb = mx.metal.device_info()["memory_size"] / 1e9
    base_model_gb = 1.4
    capacity = select_capacity(base_model_gb, device_gb,
                               num_moe_layers=num_moe_layers)

    enable_lazy_experts(model, model_path,
                        cache_capacity_per_layer=capacity,
                        predictive=True)
    mx.eval(model.parameters())

    # Memory guard
    active_gb = mx.metal.get_active_memory() / 1e9
    expert_slot_mb = 1.69
    projected_gb = active_gb + capacity * num_moe_layers * expert_slot_mb / 1024
    limit_gb = 0.85 * device_gb
    if projected_gb > limit_gb:
        max_cap = int((limit_gb - active_gb) * 1024 / (num_moe_layers * expert_slot_mb))
        capacity = (max_cap // 8) * 8
        capacity = max(capacity, 0)
        print(f"  [memory guard: reducing capacity to {capacity}]")
        enable_lazy_experts(model, model_path,
                            cache_capacity_per_layer=capacity,
                            predictive=True)
        mx.eval(model.parameters())

    t_load = time.perf_counter() - t0
    print(f"  Model load: {t_load:.1f}s ({mx.get_active_memory() / 1e9:.1f} GB)")

    # Resolve cache file path
    cache_path = None
    prepacked_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        safe_name = model_name.replace("/", "--")
        cache_path = os.path.join(cache_dir, f"{safe_name}.json")
        prepacked_path = cache_path.replace(".json", ".weights.safetensors")

    # --- Warm start paths ---
    used_saved_state = False

    if prepacked and prepacked_path and os.path.exists(prepacked_path):
        # Fastest warm start: load pre-stacked tensors directly
        t0 = time.perf_counter()
        with _with_cache_limit_zero():
            load_prepacked_weights(model, prepacked_path, model_path=model_path)
        t_upgrade = time.perf_counter() - t0
        print(f"  Prepacked load: {t_upgrade:.1f}s")

        # Delta warmup if prompt differs
        if cache_path and os.path.exists(cache_path):
            cache_state = load_cache_state(cache_path)
            saved_prompt = cache_state.get("metadata", {}).get("prompt")
            if saved_prompt and saved_prompt != prompt:
                t0 = time.perf_counter()
                with _with_cache_limit_zero():
                    fast_delta_warmup(model, tokenizer, model_path, prompt,
                                      discovery_tokens=10)
                print(f"  Delta warmup: {time.perf_counter() - t0:.1f}s")
        used_saved_state = True

    elif cache_path and os.path.exists(cache_path):
        # Standard warm start: reload from safetensors shards
        t0 = time.perf_counter()
        cache_state = load_cache_state(cache_path)
        with _with_cache_limit_zero():
            upgrade_from_saved_state(model, model_path, cache_state, capacity)
        t_upgrade = time.perf_counter() - t0
        print(f"  Cache state upgrade: {t_upgrade:.1f}s")

        saved_prompt = cache_state.get("metadata", {}).get("prompt")
        if saved_prompt and saved_prompt != prompt:
            t0 = time.perf_counter()
            with _with_cache_limit_zero():
                fast_delta_warmup(model, tokenizer, model_path, prompt,
                                  discovery_tokens=10)
            print(f"  Delta warmup: {time.perf_counter() - t0:.1f}s")
        used_saved_state = True

    else:
        # --- Cold start paths ---
        if profile_path is not None:
            # Profile-based: skip discovery entirely
            t0 = time.perf_counter()
            profile = load_universal_profile(profile_path)
            with _with_cache_limit_zero():
                upgrade_from_profile(model, model_path, capacity, profile)
            t_upgrade = time.perf_counter() - t0
            print(f"  Profile-based upgrade: {t_upgrade:.1f}s")
        else:
            # Router-only discovery (~1-2s vs ~75s full model)
            t0 = time.perf_counter()
            with _with_cache_limit_zero():
                router_only_discovery(model, tokenizer, prompt, max_tokens=10)
                upgrade_to_predictive(model, model_path, capacity)
            t_upgrade = time.perf_counter() - t0
            print(f"  Router-only discovery + upgrade: {t_upgrade:.1f}s")

    # Save state for next run
    if cache_path and not used_saved_state:
        save_cache_state(model, cache_path,
                         metadata={"prompt": prompt, "capacity": capacity})

    # Save prepacked weights for fastest warm start next time
    if prepacked and prepacked_path and not os.path.exists(prepacked_path):
        t0 = time.perf_counter()
        save_prepacked_weights(model, prepacked_path)
        print(f"  Save prepacked: {time.perf_counter() - t0:.1f}s")

    t_total = time.perf_counter() - t_total_start
    print(f"  Total startup: {t_total:.1f}s")

    return _mlx_lm.generate(model, tokenizer, prompt=prompt,
                             max_tokens=max_tokens, verbose=False)


# ---------------------------------------------------------------------------
# ML-based cache replacement (Task 5)
# ---------------------------------------------------------------------------

def dynamic_cache_update_ml(model, eviction_models, max_layer_updates=12):
    """Like dynamic_cache_update but uses ML eviction scoring.

    Instead of LCP priority, uses a tiny FFN per layer to predict eviction
    scores (approximating Belady distance).

    Args:
        eviction_models: Dict[int, nn.Module] mapping layer_idx to a trained
            FFN: input=[1/recency, freq/max_freq], output=eviction score.
            Higher score = evict first (longer predicted distance to next use).
        max_layer_updates: Max layers to perform swaps on per call.

    Returns per-layer stats list (same format as dynamic_cache_update).
    """
    stats = []
    swap_budget = max_layer_updates

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        if not isinstance(proj, PredictiveCachedSwitchLinear):
            continue
        cache = proj._cache

        if len(cache._indices_buffer) < 2:
            stats.append({"layer": i, "swaps": 0, "fallbacks": 0, "requests": 0})
            continue

        to_process = cache._indices_buffer[:-1]
        cache._indices_buffer = cache._indices_buffer[-1:]

        all_requested: set[int] = set()
        for indices in to_process:
            flat = np.asarray(indices.reshape(-1))
            all_requested |= set(int(x) for x in np.unique(flat))

        cache.step += 1
        for eid in all_requested:
            cache.frequency[eid] = cache.frequency.get(eid, 0) + 1
            cache.last_active[eid] = cache.step

        misses = all_requested - cache.cached_set
        n_requests = len(all_requested)
        n_fallbacks = len(misses)
        cache.total_requests += n_requests
        cache.total_fallbacks += n_fallbacks

        if not misses or not cache._shard_paths or swap_budget <= 0:
            stats.append({"layer": i, "swaps": 0, "fallbacks": n_fallbacks,
                         "requests": n_requests})
            continue

        ml_model = eviction_models.get(i)
        max_freq = max(cache.frequency.values()) if cache.frequency else 1

        evict_candidates = []
        for slot, eid in enumerate(cache.cached_ids):
            if eid in all_requested or eid in cache.pinned_set:
                continue
            if ml_model is not None:
                recency = cache.step - cache.last_active.get(eid, 0)
                freq = cache.frequency.get(eid, 0)
                features = mx.array([[1.0 / max(recency, 1), freq / max(max_freq, 1)]])
                score = float(ml_model(features).item())
            else:
                score = -cache._lcp_priority(eid)
            evict_candidates.append((score, slot, eid))

        evict_candidates.sort(reverse=True)

        swaps: list[tuple[int, int, int]] = []
        for new_eid in sorted(misses):
            if not evict_candidates:
                break
            _, slot, old_eid = evict_candidates.pop(0)
            swaps.append((slot, old_eid, new_eid))

        MAX_SWAPS = 10
        swaps = swaps[:MAX_SWAPS]

        if not swaps:
            stats.append({"layer": i, "swaps": 0, "fallbacks": n_fallbacks,
                         "requests": n_requests})
            continue

        new_eids = mx.array([new_eid for _, _, new_eid in swaps])
        slot_indices = mx.array([slot for slot, _, _ in swaps])
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            shard_path = cache._shard_paths[proj_name]
            key_prefix = cache._key_prefixes[proj_name]
            shard = mx.load(shard_path)
            new_w, new_s, new_b = _load_proj_experts(shard, key_prefix, new_eids,
                                                      shard_map=cache._shard_map)
            del shard

            if new_b is None:
                mx.eval(new_w, new_s)
            else:
                mx.eval(new_w, new_s, new_b)

            w = cache.weights.pop(proj_name)
            w[slot_indices] = new_w
            cache.weights[proj_name] = w

            s = cache.scales.pop(proj_name)
            s[slot_indices] = new_s
            cache.scales[proj_name] = s

            if cache.biases[proj_name] is not None and new_b is not None:
                b = cache.biases.pop(proj_name)
                b[slot_indices] = new_b
                cache.biases[proj_name] = b

        mx.clear_cache()

        for slot, old_eid, new_eid in swaps:
            cache.cached_set.discard(old_eid)
            cache.cached_set.add(new_eid)
            cache.cached_ids[slot] = new_eid
            cache.frequency.pop(old_eid, None)
            cache.last_active.pop(old_eid, None)

        lookup_np = np.zeros(cache.num_experts, dtype=np.int32)
        for slot, eid in enumerate(cache.cached_ids):
            lookup_np[eid] = slot
        cache.lookup = mx.array(lookup_np)
        mx.eval(cache.lookup)

        swap_budget -= 1
        stats.append({"layer": i, "swaps": len(swaps), "fallbacks": n_fallbacks,
                     "requests": n_requests})

    return stats
