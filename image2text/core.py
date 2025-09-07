# image2text/core.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
from image2text.backends import LlavaRunner

def generate(
    images: List[Union[str, Path, Image.Image]],
    prompt: Optional[str] = None,
    *,
    model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    device: Optional[Union[int, str]] = None,
    dtype: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    runner = LlavaRunner(model_id=model_id, device=device, dtype=dtype)
    return runner.describe(
        images=images,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
