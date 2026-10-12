"""Framework-free image handling for Qwen3.5's vision tower (shared by torch and MLX).

An image in the state becomes `<|vision_start|>` + N x `<|image_pad|>` + `<|vision_end|>`
in the token stream, where N = (h / 32) * (w / 32) after `smart_resize` (16 px patches,
2x2 merged into one token). `pixel_patches` reproduces the Hugging Face Qwen2-VL image
processor (PIL backend): bicubic resize, scale to [-1, 1], (merge-major) patch flattening
with the temporal axis duplicated, giving [h/16 * w/16, 3 * 2 * 16 * 16] float32.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PATCH = 16
MERGE = 2
TEMPORAL = 2
FACTOR = PATCH * MERGE
MIN_PIXELS = 65536                  # preprocessor_config "shortest_edge"
MAX_PIXELS = 1024 * FACTOR * FACTOR  # our cap: at most 1024 image tokens per image

# Qwen3.5 special tokens (config.json)
VISION_START, VISION_END, IMAGE_PAD = 248053, 248054, 248056


@dataclass(frozen=True)
class ImageRef:
    """An image in the state: a `data:` URI, or a file path (relative paths resolve against
    `base` — the directory of the JSONL file it came from)."""

    source: str
    base: str = ""

    def open(self):
        from PIL import Image

        if self.source.startswith("data:"):
            raw = base64.b64decode(self.source.split(",", 1)[1])
            img = Image.open(io.BytesIO(raw))
        else:
            img = Image.open(Path(self.base) / self.source)
        return img.convert("RGB")


def smart_resize(height: int, width: int, factor: int = FACTOR, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Same as transformers' qwen2_vl `smart_resize`."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio must be below 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_grid(size: tuple[int, int], max_pixels: int = MAX_PIXELS) -> tuple[int, int, int]:
    """(t, h, w) patch grid for a PIL-style (width, height) size."""
    w, h = size
    rh, rw = smart_resize(h, w, max_pixels=max_pixels)
    return 1, rh // PATCH, rw // PATCH


def n_image_tokens(grid: tuple[int, int, int]) -> int:
    t, h, w = grid
    return t * h * w // (MERGE * MERGE)


def image_tokens(grid: tuple[int, int, int]) -> list[int]:
    return [VISION_START] + [IMAGE_PAD] * n_image_tokens(grid) + [VISION_END]


def pixel_patches(img, grid: tuple[int, int, int]) -> np.ndarray:
    """[h * w, 3 * TEMPORAL * PATCH * PATCH] float32 for a PIL RGB image."""
    from PIL import Image

    _, gh, gw = grid
    x = np.asarray(img.resize((gw * PATCH, gh * PATCH), Image.Resampling.BICUBIC), dtype=np.float32)
    x = (x / 255.0 - 0.5) / 0.5
    x = x.transpose(2, 0, 1)  # C, H, W
    x = x.reshape(3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.transpose(1, 4, 2, 5, 0, 3, 6)  # gh', gw', mh, mw, C, ph, pw
    x = np.broadcast_to(x[:, :, :, :, :, None], (*x.shape[:5], TEMPORAL, PATCH, PATCH))
    return np.ascontiguousarray(x.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH))
