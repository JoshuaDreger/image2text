# image2text/api.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import os, io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from image2text.core import generate

app = FastAPI(title="image2text API", version="1.0")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = os.getenv("I2T_MODEL", "llava-hf/llava-onevision-qwen2-7b-ov-hf")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/model")
def model():
    return {"model": DEFAULT_MODEL}

@app.post("/api/describe")
async def describe(
    files: List[UploadFile] = File(..., description="One or more images"),
    prompt: Optional[str] = Form(None),
    max_new_tokens: int = Form(256),
    temperature: float = Form(0.2),
) -> JSONResponse:
    if not files:
        raise HTTPException(422, "Upload one or more images.")
    pil_images = []
    for f in files:
        data = await f.read()
        try:
            pil_images.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception:
            raise HTTPException(400, f"Unsupported image: {f.filename}")

    try:
        text = generate(
            images=pil_images,
            prompt=prompt,
            model_id=DEFAULT_MODEL,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    # Save result to /data
    stem = Path(files[0].filename).stem if files[0].filename else "image"
    out_path = OUTPUT_DIR / f"{stem}.txt"
    out_path.write_text(text, encoding="utf-8")

    return JSONResponse({"status": "ok", "output": str(out_path), "text": text})
