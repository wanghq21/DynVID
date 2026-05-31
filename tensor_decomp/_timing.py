"""
_timing.py — 极简全局计时器，供 modeling_qwen3_vl / compression_unified / qwen3_vl 使用。

用法：
  from . import _timing
  _timing.add("vision_encoder_ms", elapsed_ms)
  _timing.add("compression_ms", elapsed_ms)

  # 最后读取
  result = _timing.get_all()  # {"vision_encoder_ms": 120.5, "compression_ms": 45.2, ...}
  _timing.reset()
"""

import threading

_lock = threading.Lock()
_data: dict = {}


def add(key: str, value: float) -> None:
    """累加一个计时值（毫秒）。"""
    with _lock:
        _data[key] = _data.get(key, 0.0) + float(value)


def get_all() -> dict:
    """返回所有累计值的拷贝。"""
    with _lock:
        return dict(_data)


def reset() -> None:
    """清空所有累计值。"""
    with _lock:
        _data.clear()
