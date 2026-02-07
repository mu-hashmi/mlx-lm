# Copyright © 2023-2025 Apple Inc.

import os
import time
from pathlib import Path

import mlx.core as mx

from .loading import (
    _find_switch_mlp,
    _detect_num_experts,
    select_capacity,
    _with_cache_limit_zero,
)
from .core import enable_lazy_experts, upgrade_to_predictive
from .warmup import fast_delta_warmup
from .discovery import router_only_discovery
from .persistence import (
    save_cache_state,
    load_cache_state,
    save_prepacked_weights,
    load_prepacked_weights,
    upgrade_from_saved_state,
    load_universal_profile,
    upgrade_from_profile,
)


def flash_generate(model_name: str, prompt: str, max_tokens: int = 200,
                   cache_dir: str | None = None,
                   profile_path: str | None = None,
                   prepacked: bool = True) -> str:
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

    device_gb = mx.device_info()["memory_size"] / 1e9
    base_model_gb = 1.4
    capacity = select_capacity(base_model_gb, device_gb,
                               num_moe_layers=num_moe_layers)

    enable_lazy_experts(model, model_path,
                        cache_capacity_per_layer=capacity,
                        predictive=True)
    mx.eval(model.parameters())

    # Memory guard
    active_gb = mx.get_active_memory() / 1e9
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

    # 256 MB cache floor during warmup: retains small buffer cache for
    # intermediate reuse, improving generation throughput by ~14% vs cache=0.
    _WARMUP_CACHE = 256 * 1024 * 1024

    used_saved_state = False

    if prepacked and prepacked_path and os.path.exists(prepacked_path):
        t0 = time.perf_counter()
        with _with_cache_limit_zero(_WARMUP_CACHE):
            load_prepacked_weights(model, prepacked_path, model_path=model_path)
        t_upgrade = time.perf_counter() - t0
        print(f"  Prepacked load: {t_upgrade:.1f}s")

        if cache_path and os.path.exists(cache_path):
            cache_state = load_cache_state(cache_path)
            saved_prompt = cache_state.get("metadata", {}).get("prompt")
            if saved_prompt and saved_prompt != prompt:
                t0 = time.perf_counter()
                with _with_cache_limit_zero(_WARMUP_CACHE):
                    fast_delta_warmup(model, tokenizer, model_path, prompt,
                                      discovery_tokens=10)
                print(f"  Delta warmup: {time.perf_counter() - t0:.1f}s")
        used_saved_state = True

    elif cache_path and os.path.exists(cache_path):
        t0 = time.perf_counter()
        cache_state = load_cache_state(cache_path)
        with _with_cache_limit_zero(_WARMUP_CACHE):
            upgrade_from_saved_state(model, model_path, cache_state, capacity)
        t_upgrade = time.perf_counter() - t0
        print(f"  Cache state upgrade: {t_upgrade:.1f}s")

        saved_prompt = cache_state.get("metadata", {}).get("prompt")
        if saved_prompt and saved_prompt != prompt:
            t0 = time.perf_counter()
            with _with_cache_limit_zero(_WARMUP_CACHE):
                fast_delta_warmup(model, tokenizer, model_path, prompt,
                                  discovery_tokens=10)
            print(f"  Delta warmup: {time.perf_counter() - t0:.1f}s")
        used_saved_state = True

    else:
        if profile_path is not None:
            t0 = time.perf_counter()
            profile = load_universal_profile(profile_path)
            with _with_cache_limit_zero(_WARMUP_CACHE):
                upgrade_from_profile(model, model_path, capacity, profile)
            t_upgrade = time.perf_counter() - t0
            print(f"  Profile-based upgrade: {t_upgrade:.1f}s")
        else:
            t0 = time.perf_counter()
            with _with_cache_limit_zero(_WARMUP_CACHE):
                router_only_discovery(model, tokenizer, prompt, max_tokens=10)
                upgrade_to_predictive(model, model_path, capacity)
            t_upgrade = time.perf_counter() - t0
            print(f"  Router-only discovery + upgrade: {t_upgrade:.1f}s")

    if cache_path and not used_saved_state:
        save_cache_state(model, cache_path,
                         metadata={"prompt": prompt, "capacity": capacity})

    if prepacked and prepacked_path and not os.path.exists(prepacked_path):
        t0 = time.perf_counter()
        save_prepacked_weights(model, prepacked_path)
        print(f"  Save prepacked: {time.perf_counter() - t0:.1f}s")

    t_total = time.perf_counter() - t_total_start
    print(f"  Total startup: {t_total:.1f}s")

    if hasattr(mx, "set_wired_limit"):
        active = mx.get_active_memory()
        limit = int(mx.device_info()["memory_size"] * 0.75)
        wired = min(active, limit)
        mx.set_wired_limit(wired)
        print(f"  Wired {wired / 1e9:.1f} GB in residency set")

    return _mlx_lm.generate(model, tokenizer, prompt=prompt,
                             max_tokens=max_tokens, verbose=False)
