# image2text/web_app.py
import os, sys, json, tempfile
from pathlib import Path
import streamlit as st
from PIL import Image
from image2text.core import generate

def resolve_output_dir() -> Path:
    cand = os.getenv("OUTPUT_DIR")
    if cand:
        p = Path(cand).expanduser().resolve()
    else:
        # default to ./data inside the project
        p = (Path(__file__).parent / ".." / "data").resolve()
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except PermissionError:
        # last-ditch fallback to user temp dir
        t = Path(tempfile.gettempdir()) / "image2text"
        t.mkdir(parents=True, exist_ok=True)
        return t

OUTPUT_DIR = resolve_output_dir()
MODEL_ID = os.getenv("I2T_MODEL", "llava-hf/llava-onevision-qwen2-7b-ov-hf")

st.set_page_config(page_title="image2text (LLaVA-OV)", page_icon="🖼️")

# --- Model selection UI ---
# Pre-populate with smaller LLaVA-OneVision variants that are friendlier for laptops.
# You can also pick "Custom…" to type any Hugging Face model id.
DEFAULT_MODEL = MODEL_ID
MODEL_CANDIDATES = {
    "LLaVA-OV Qwen2 0.5B (tiny, CPU-friendly)": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "LLaVA-OV Qwen2 1.5B (small)": "llava-hf/llava-onevision-qwen2-1.5b-ov-hf",
    "LLaVA-OV Qwen2 7B (default)": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "Custom…": "__custom__",
}
# Try to set the selectbox default to the env model if it matches one of the known ids.
def _initial_label_for_model(mid: str) -> str:
    for label, val in MODEL_CANDIDATES.items():
        if val == mid:
            return label
    return "Custom…"

with st.sidebar:
    st.header("⚙️ Settings")
    selected_label = st.selectbox(
        "Vision-Language Model",
        list(MODEL_CANDIDATES.keys()),
        index=list(MODEL_CANDIDATES.keys()).index(_initial_label_for_model(DEFAULT_MODEL)),
        help="Pick a smaller model if the 7B one is too heavy for your machine.",
    )
    if MODEL_CANDIDATES[selected_label] == "__custom__":
        MODEL_ID = st.text_input(
            "Hugging Face model id",
            value=DEFAULT_MODEL,
            placeholder="e.g. llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        )
    else:
        MODEL_ID = MODEL_CANDIDATES[selected_label]
    st.caption(f"Selected model: `{MODEL_ID}`")

    # --- Device selection UI ---
    DEVICE_OPTIONS = {
        "Auto (prefer GPU)": None,   # let backend decide (e.g., device_map='auto')
        "GPU (CUDA)": "cuda",        # force CUDA path
        "CPU": "cpu",                # force CPU
    }
    device_choice = st.selectbox(
        "Compute device",
        list(DEVICE_OPTIONS.keys()),
        index=0,
        help="Choose where to run inference. 'Auto' will try GPU if available, otherwise CPU."
    )
    DEVICE = DEVICE_OPTIONS[device_choice]
    st.caption(f"Using device: {device_choice}")

st.title("Image → Text (LLaVA-OneVision)")
st.caption(f"Working dir: `{Path.cwd()}` · Python: `{sys.executable}` · OUTPUT_DIR: `{OUTPUT_DIR}`")

with st.expander("Upload image(s)"):
    uploaded = st.file_uploader(
        "Images (PNG/JPG/WebP); you can select multiple",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

prompt = st.text_area("Optional prompt / question", placeholder="e.g., 'Describe the chart in detail.'")
col_a, col_b = st.columns(2)
with col_a:
    max_new_tokens = st.slider("Max new tokens", 32, 1024, 256, step=32)
with col_b:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.05)

if st.button("Generate") and uploaded:
    images = []
    for f in uploaded:
        images.append(Image.open(f).convert("RGB"))
    try:
        text = generate(
            images,
            prompt=prompt,
            model_id=MODEL_ID,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=DEVICE,  # <-- pass user's choice
        )

        # Save + show
        stem = Path(uploaded[0].name).stem
        out_txt = OUTPUT_DIR / f"{stem}.txt"
        out_txt.write_text(text, encoding="utf-8")

        st.success(f"Saved → {out_txt}")
        st.code(text)
        st.download_button("Download .txt", data=text.encode("utf-8"), file_name=out_txt.name)
    except Exception as e:
        st.error(f"Failed: {e}")
