"""
Lightweight efficiency recorder for DySeg-DPP / lmms-eval experiments.

The recorder is intentionally no-op when disabled so normal evaluation logic is
unchanged unless enable_efficiency=True is explicitly passed from model_args.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover - keep import safe in non-torch contexts
    torch = None

_TLS = threading.local()
_PROCESS_METRICS: Dict[str, Any] = {}
_PROCESS_LOCK = threading.Lock()


class EfficiencyRecorder:
    """Per-process JSONL recorder used by lmms-eval model wrappers."""

    def __init__(
        self,
        enabled: bool = False,
        output_path: Optional[str] = None,
        backend: str = "baseline",
        model_name: str = "unknown",
        rank: Optional[int] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_path = output_path
        self.backend = backend
        self.model_name = model_name
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) if rank is None else int(rank)
        self._sample: Dict[str, Any] = {}
        self._start_time: Optional[float] = None

    def reset_sample(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        self._sample = {
            "backend": self.backend,
            "model": self.model_name,
            "rank": self.rank,
        }
        if extra:
            self._sample.update(_json_safe_dict(extra))
        self._start_time = time.perf_counter()
        set_current_recorder(self)

    def add_time(self, key: str, ms: float) -> None:
        if not self.enabled:
            return
        try:
            value = float(ms)
        except Exception:
            return
        self._sample[key] = float(self._sample.get(key, 0.0)) + value

    def set_metric(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._sample[key] = _json_safe(value)

    def update(self, metrics: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._sample.update(_json_safe_dict(metrics))

    def finalize(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        if self._start_time is not None and "generate_ms" not in self._sample:
            cuda_synchronize_if_needed()
            self._sample["generate_ms"] = (time.perf_counter() - self._start_time) * 1000.0
        if extra:
            self._sample.update(_json_safe_dict(extra))
        self._write(self._sample)
        clear_current_recorder(self)
        self._sample = {}
        self._start_time = None

    def _write(self, record: Dict[str, Any]) -> None:
        if not self.output_path:
            return
        path = Path(self.output_path)
        if self.rank is not None:
            path = path.with_name(f"{path.stem}_rank{self.rank}{path.suffix or '.jsonl'}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def set_current_recorder(recorder: Optional[EfficiencyRecorder]) -> None:
    _TLS.recorder = recorder


def get_current_recorder() -> Optional[EfficiencyRecorder]:
    return getattr(_TLS, "recorder", None)


def clear_current_recorder(recorder: Optional[EfficiencyRecorder] = None) -> None:
    current = get_current_recorder()
    if recorder is None or current is recorder:
        _TLS.recorder = None


def is_enabled() -> bool:
    return os.environ.get("QWEN3_VL_EFFICIENCY_OUTPUT", "") != "" or os.environ.get("QWEN3_VL_ENABLE_EFFICIENCY", "").lower() in {"1", "true", "yes", "y"}


def cuda_synchronize_if_needed(device=None) -> None:
    if torch is None or not torch.cuda.is_available():
        return
    try:
        if device is not None and hasattr(device, "type") and device.type != "cuda":
            return
        torch.cuda.synchronize(device=device if device is not None else None)
    except TypeError:
        torch.cuda.synchronize()


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def add_time(key: str, ms: float) -> None:
    recorder = get_current_recorder()
    try:
        value = float(ms)
    except Exception:
        return
    with _PROCESS_LOCK:
        _PROCESS_METRICS[key] = float(_PROCESS_METRICS.get(key, 0.0)) + value
    if recorder is not None:
        recorder.add_time(key, value)


def set_metric(key: str, value: Any) -> None:
    safe_value = _json_safe(value)
    with _PROCESS_LOCK:
        # num_videos / input_tokens / output_tokens 需要跨 batch 累加，
        # 这样 fallback record 才能得到总量用于计算 per-video 平均。
        if key in {"num_videos", "input_tokens", "output_tokens"}:
            if safe_value is not None:
                try:
                    prev = _PROCESS_METRICS.get(key)
                    prev_f = float(prev) if prev is not None else 0.0
                    _PROCESS_METRICS[key] = prev_f + float(safe_value)
                except Exception:
                    _PROCESS_METRICS[key] = safe_value
            # value 为 None 时不累加、也不覆盖（保留已有累加值）
        else:
            _PROCESS_METRICS[key] = safe_value
    recorder = get_current_recorder()
    if recorder is not None:
        recorder.set_metric(key, safe_value)


def update(metrics: Dict[str, Any]) -> None:
    safe_metrics = _json_safe_dict(metrics)
    with _PROCESS_LOCK:
        _PROCESS_METRICS.update(safe_metrics)
    recorder = get_current_recorder()
    if recorder is not None:
        recorder.update(safe_metrics)


def get_process_metrics() -> Dict[str, Any]:
    with _PROCESS_LOCK:
        return dict(_PROCESS_METRICS)


def clear_process_metrics() -> None:
    with _PROCESS_LOCK:
        _PROCESS_METRICS.clear()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if torch is not None and hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return _json_safe_dict(value)
    return str(value)


def _json_safe_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): _json_safe(v) for k, v in data.items()}