import json
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .models.switch_layers import QuantizedSwitchLinear


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
                 cache: ExpertCache):
        super().__init__()
        self._shard_path = shard_path
        self._key_prefix = key_prefix
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
            w_batch = shard[f"{self._key_prefix}.weight"][miss_arr]
            s_batch = shard[f"{self._key_prefix}.scales"][miss_arr]
            biases_key = f"{self._key_prefix}.biases"
            b_batch = shard[biases_key][miss_arr] if biases_key in shard else None
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
                 '_shard_paths', '_key_prefixes',
                 'total_requests', 'total_fallbacks')

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
        self.total_requests: int = 0
        self.total_fallbacks: int = 0

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

        # Find coldest cached experts to evict (exclude currently-requested)
        evict_candidates = [
            (self._lcp_priority(eid), slot, eid)
            for slot, eid in enumerate(self.cached_ids)
            if eid not in all_requested
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
            new_w = shard[f"{key_prefix}.weight"][new_eids]
            new_s = shard[f"{key_prefix}.scales"][new_eids]
            biases_key = f"{key_prefix}.biases"
            new_b = shard[biases_key][new_eids] if biases_key in shard else None
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
    """Read model.safetensors.index.json and return {key: absolute_shard_path}."""
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    return {key: str(model_path / shard) for key, shard in weight_map.items()}


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
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        switch = layer.mlp.switch_mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(switch, name)
            if not isinstance(orig, QuantizedSwitchLinear):
                continue
            key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            replacement = LazyQuantizedSwitchLinear(
                shard_path=shard_path,
                key_prefix=key_prefix,
                group_size=orig.group_size,
                bits=orig.bits,
                mode=orig.mode,
            )
            setattr(switch, name, replacement)
            replaced += 1
    return replaced


def _enable_cached(model, shard_map: dict, capacity: int) -> int:
    replaced = 0
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        switch = layer.mlp.switch_mlp
        layer_cache = ExpertCache(capacity)
        for name in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(switch, name)
            if not isinstance(orig, QuantizedSwitchLinear):
                continue
            key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            replacement = CachedQuantizedSwitchLinear(
                shard_path=shard_path,
                key_prefix=key_prefix,
                group_size=orig.group_size,
                bits=orig.bits,
                mode=orig.mode,
                proj_name=name,
                cache=layer_cache,
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
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        switch = layer.mlp.switch_mlp
        first = getattr(switch, "gate_proj")
        if not isinstance(first, (PredictiveCachedSwitchLinear, SyncPredictiveCachedSwitchLinear)):
            continue

        layer_cache = ExpertCache(capacity)
        for name in ("gate_proj", "up_proj", "down_proj"):
            pred_mod = getattr(switch, name)
            key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]
            replacement = CachedQuantizedSwitchLinear(
                shard_path=shard_path,
                key_prefix=key_prefix,
                group_size=pred_mod.group_size,
                bits=pred_mod.bits,
                mode=pred_mod.mode,
                proj_name=name,
                cache=layer_cache,
            )
            setattr(switch, name, replacement)
            reset += 1

    mx.clear_cache()
    return reset


def upgrade_to_predictive(model, model_path: Path, capacity,
                          sync: bool = False) -> int:
    """Harvest Phase 2 LCP caches into zero-eval predictive tensors.

    Call this after running warmup generation with Phase 2 (CachedQuantizedSwitchLinear).
    Harvests discovered experts from LCP caches, fills remaining capacity from disk,
    then swaps to PredictiveCachedSwitchLinear for zero-eval forward pass.

    Args:
        capacity: int for uniform capacity, or list[int] for per-MoE-layer capacities.
        sync: If True, use SyncPredictiveCachedSwitchLinear (adds mx.eval per layer,
              for benchmarking the sync-point hypothesis).

    Returns number of modules upgraded.
    """
    model_path = Path(model_path)
    shard_map = _build_shard_map(model_path)
    per_layer_caps = isinstance(capacity, (list, tuple))

    upgraded = 0
    moe_idx = 0
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        switch = layer.mlp.switch_mlp

        # Get the Phase 2 module and its LCP cache
        first_proj = getattr(switch, "gate_proj")
        if not isinstance(first_proj, CachedQuantizedSwitchLinear):
            continue
        lcp_cache = first_proj._cache
        num_experts = 512  # Qwen3-Coder-Next
        C = min(capacity[moe_idx] if per_layer_caps else capacity, num_experts)
        moe_idx += 1

        # Harvest expert IDs discovered by LCP, sorted by priority (best first)
        discovered = sorted(
            lcp_cache.entries.keys(),
            key=lambda eid: lcp_cache._priority(eid),
            reverse=True,
        )[:C]
        discovered_set = set(discovered)

        # Fill remaining capacity with sequential IDs not already discovered
        filler = []
        for eid in range(num_experts):
            if len(discovered) + len(filler) >= C:
                break
            if eid not in discovered_set:
                filler.append(eid)
        cached_ids = list(discovered) + filler

        pred_cache = PredictiveExpertCache(C, num_experts)

        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mod = getattr(switch, name)
            key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"
            shard_path = shard_map[f"{key_prefix}.weight"]

            # Collect tensors: harvested from LCP cache + loaded from disk
            ws, ss, bs = [], [], []
            to_load = []  # (slot_index, expert_id) for disk loads
            has_bias = None

            for slot, eid in enumerate(cached_ids):
                cached = lcp_cache.lookup(eid, name)
                if cached is not None:
                    w, s, b = cached
                    ws.append(w)
                    ss.append(s)
                    if has_bias is None:
                        has_bias = b is not None
                    if has_bias:
                        bs.append(b)
                else:
                    to_load.append((slot, eid))
                    ws.append(None)
                    ss.append(None)
                    if has_bias is None:
                        has_bias = False  # determined by first non-None entry
                    if has_bias:
                        bs.append(None)

            # Batch-load missing experts from disk
            if to_load:
                load_ids = mx.array([eid for _, eid in to_load])
                shard = mx.load(shard_path)
                w_batch = shard[f"{key_prefix}.weight"][load_ids]
                s_batch = shard[f"{key_prefix}.scales"][load_ids]
                biases_key = f"{key_prefix}.biases"
                b_batch = shard[biases_key][load_ids] if biases_key in shard else None
                if b_batch is None:
                    mx.eval(w_batch, s_batch)
                else:
                    mx.eval(w_batch, s_batch, b_batch)
                    if has_bias is None:
                        has_bias = True

                for j, (slot, _) in enumerate(to_load):
                    ws[slot] = w_batch[j]
                    ss[slot] = s_batch[j]
                    if has_bias and b_batch is not None:
                        bs[slot] = b_batch[j]

            pred_cache.weights[name] = mx.stack(ws)
            pred_cache.scales[name] = mx.stack(ss)
            pred_cache.biases[name] = mx.stack(bs) if has_bias else None

        # Store shard info for dynamic updates
        for name in ("gate_proj", "up_proj", "down_proj"):
            key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"
            pred_cache._shard_paths[name] = shard_map[f"{key_prefix}.weight"]
            pred_cache._key_prefixes[name] = key_prefix

        # Build lookup and install modules
        pred_cache.build_lookup(cached_ids)
        mx.eval(pred_cache.lookup)

        cls = SyncPredictiveCachedSwitchLinear if sync else PredictiveCachedSwitchLinear
        for name in ("gate_proj", "up_proj", "down_proj"):
            phase2_mod = getattr(switch, name)
            replacement = cls(
                group_size=phase2_mod.group_size,
                bits=phase2_mod.bits,
                mode=phase2_mod.mode,
                proj_name=name,
                cache=pred_cache,
            )
            setattr(switch, name, replacement)
            upgraded += 1

        # Clear LCP cache to free duplicated tensors
        lcp_cache.entries.clear()
        lcp_cache.frequency.clear()
        lcp_cache.last_active.clear()

        print(f"  Layer {i}: {len(discovered)} discovered + {len(filler)} filler "
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
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        switch = layer.mlp.switch_mlp
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
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        switch = layer.mlp.switch_mlp
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
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
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
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
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

    # Step 3: Swap missing experts into cache, one layer at a time
    total_swaps = 0
    total_missing = 0
    per_layer_stats = []

    for i, info in layer_info.items():
        missing = info["missing"]
        cold = list(info["cold"])
        cache = info["cache"]
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

        new_eids = mx.array([new_eid for _, _, new_eid in swaps])

        # Batch-load new experts: group projections by shard to minimize disk reads
        shard_groups = {}
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            sp = cache._shard_paths[proj_name]
            shard_groups.setdefault(sp, []).append(
                (proj_name, cache._key_prefixes[proj_name]))

        new_experts = {}
        to_eval_new = []
        for shard_path, group in shard_groups.items():
            shard = mx.load(shard_path)
            for proj_name, key_prefix in group:
                new_w = shard[f"{key_prefix}.weight"][new_eids]
                new_s = shard[f"{key_prefix}.scales"][new_eids]
                biases_key = f"{key_prefix}.biases"
                new_b = shard[biases_key][new_eids] if biases_key in shard else None
                new_experts[proj_name] = (new_w, new_s, new_b)
                to_eval_new.extend([new_w, new_s])
                if new_b is not None:
                    to_eval_new.append(new_b)
            del shard

        # One eval per layer for all new experts (frees shard references)
        mx.eval(*to_eval_new)

        # Scatter update: write new experts directly into stacked tensors.
        # pop() drops the dict ref so the buffer is singly-referenced,
        # enabling MLX's buffer donation (zero-copy reuse) in the scatter op.
        slot_indices = mx.array([slot for slot, _, _ in swaps])
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            new_w, new_s, new_b = new_experts[proj_name]

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

        del new_experts

        # Update cached_ids and rebuild lookup table
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

        # Eval rebuilt tensors + lookup for this layer to free old tensor references.
        # Without this, all 48 layers' old+new tensors coexist → OOM.
        to_eval = [cache.lookup]
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            to_eval.append(cache.weights[proj_name])
            to_eval.append(cache.scales[proj_name])
            if cache.biases[proj_name] is not None:
                to_eval.append(cache.biases[proj_name])
        mx.eval(*to_eval)

        total_swaps += len(swaps)
        per_layer_stats.append({
            "layer": i, "missing": len(missing), "swapped": len(swaps),
        })

    t_rebuild = time.perf_counter() - t1

    # Reset fallback counters for clean generation stats
    for layer in model.layers:
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
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
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "gate_proj")
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
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        proj = getattr(layer.mlp.switch_mlp, "up_proj", None)
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
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
            continue
        switch = layer.mlp.switch_mlp
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
