# image2text/backends.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
from transformers import pipeline

class LlavaRunner:
    def __init__(
        self,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        device: Optional[Union[int, str]] = None,   # e.g. 0, "cuda", or "cpu"
        dtype: Optional[str] = None,                # e.g. "bfloat16" or "float16"
    ):
        kwargs = {"model": model_id}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        # HF added unified "image-text-to-text" task for LLaVA-OV
        self.pipe = pipeline("image-text-to-text", **kwargs)

    def describe(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        # messages format lets you mix system/user text with one or many images
        content = []
        if prompt:
            content.append({"type": "text", "text": prompt})
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            content.append({"type": "image", "image": img})

        messages = [{"role": "user", "content": content}]
        out = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
        )
        # pipeline returns a list of dicts with "generated_text"
        return out[0]["generated_text"].strip()
