#!/usr/bin/env python3
"""
KYC Portrait Generator – Fixed & Weaponised
Uses stable-diffusion-img2img for style transfer, briaai/rmbg-1.4 for background removal,
and stable-diffusion-inpainting for KYC markers.
"""

import asyncio
import io
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

import replicate
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# ---------- CONFIG ----------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_BATCH_PHOTOS = int(os.getenv("MAX_BATCH_PHOTOS", "20"))
OUTPUT_WIDTH = 1030
OUTPUT_HEIGHT = 1536
GREY_RGB = (128, 128, 128)
GPU_MAX_PIXELS = 1_800_000
CODEFORMER_FIDELITY = 0.98

MIN_SEC_BETWEEN = 15
MAX_RETRIES = 6
_req_lock = threading.Lock()
_last_pred = 0.0

# Prompts that burn the target look into the image
STYLE_PROMPT = (
    "A beautiful Indian woman wearing light grey dupatta with heavy gold embroidery "
    "floral paisley, matching grey saree, sindoor in hair, red bindi, gold nose stud, "
    "black-red beaded necklace, ornate gold pendant, studio portrait, grey background, "
    "hyper-realistic, 8k, sharp focus, no watermark"
)
NEG_PROMPT = "watermark, logo, text, cartoon, 3d, blurry, ugly, blemish, scar"

INPAINT_PROMPTS = {
    "bindi": "bright red bindi on forehead, realistic",
    "sindoor": "thick red sindoor in hair parting, natural",
    "nose_stud": "small gold nose stud in left nostril",
    "necklace": "black red beaded necklace, ornate gold pendant with white stone",
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("kyc-bot")
if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


# ---------- HELPERS ----------
def gpu_safe(input_path, output_path, max_px=GPU_MAX_PIXELS):
    with Image.open(input_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        if w * h > max_px:
            scale = (max_px / (w * h)) ** 0.5
            im = im.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.Resampling.LANCZOS)
        im.save(output_path, "JPEG", quality=95, optimize=True)
    return output_path

def out_to_bytes(o):
    if isinstance(o, (bytes, bytearray)): return bytes(o)
    if hasattr(o, "read"): return o.read()
    if isinstance(o, (list, tuple)) and o: return out_to_bytes(o[0])
    if isinstance(o, str) and o.startswith("http"):
        import requests; r = requests.get(o, timeout=180); r.raise_for_status(); return r.content
    raise RuntimeError(f"Bad output type: {type(o)}")

def is_rate_limit(exc):
    return (getattr(exc, "status", None) == 429 or "429" in str(exc).lower())

def retry_after(exc, fallback):
    resp = getattr(exc, "response", None)
    val = (resp.headers or {}).get("Retry-After") if resp else None
    try: return max(float(val), 1.0) if val else fallback
    except: return fallback

def run_model(model_ref, image_path, extra_inputs, image_field="image"):
    global _last_pred
    with _req_lock:
        for attempt in range(MAX_RETRIES + 1):
            gap = MIN_SEC_BETWEEN - (time.monotonic() - _last_pred)
            if gap > 0: time.sleep(gap)
            try:
                with open(image_path, "rb") as f:
                    payload = dict(extra_inputs)
                    payload[image_field] = f
                    _last_pred = time.monotonic()
                    result = replicate.run(model_ref, input=payload)
                return out_to_bytes(result)
            except Exception as e:
                if not is_rate_limit(e) or attempt >= MAX_RETRIES: raise
                sleep = retry_after(e, min(20*(2**attempt), 180))
                log.warning(f"Rate-limit, retry in {sleep}s")
                time.sleep(sleep)
    raise RuntimeError("All retries exhausted")


# ---------- MODELS (PUBLIC & ACCESSIBLE) ----------
# Face restoration – sczhou/codeformer (public)
def restore_face(file_path):
    return run_model(
        "sczhou/codeformer:cc4956dd26fa5a7185d5660cc9100fab1b8070a1d1654a8bb5eb6d443b020bb2",
        file_path, {"codeformer_fidelity": CODEFORMER_FIDELITY, "background_enhance": False,
                     "face_upsample": True, "upscale": 2}
    )

# Upscale – nightmareai/real-esrgan (public, no version needed)
def upscale(file_path):
    return run_model("nightmareai/real-esrgan", file_path, {"scale": 4, "face_enhance": False})

# Style transfer – stability-ai/stable-diffusion-img2img (public, no version)
def style_transfer(file_path):
    return run_model(
        "stability-ai/stable-diffusion-img2img",
        file_path,
        {
            "prompt": STYLE_PROMPT,
            "negative_prompt": NEG_PROMPT,
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "strength": 0.6,   # keep shape, change style
        }
    )

# Inpainting – stability-ai/stable-diffusion-inpainting (public)
INPAINT_MODEL = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc44a2809b8e000636d59b12a3ad3761f74ceee6df"

def inpaint_region(image_bytes, mask_rects, prompt):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img_path = d / "img.png"; img_path.write_bytes(image_bytes)
        with Image.open(img_path) as im:
            mask = Image.new("L", im.size, 0)
            draw = ImageDraw.Draw(mask)
            for rect in mask_rects: draw.rectangle(rect, fill=255)
            mask_path = d / "mask.png"; mask.save(mask_path)
        return run_model(
            INPAINT_MODEL,
            img_path,
            {
                "mask": open(mask_path, "rb"),
                "prompt": prompt + ", realistic skin, high detail",
                "negative_prompt": NEG_PROMPT,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
            }
        )

def add_markers(img_bytes):
    # work on 768x1024 for consistent coordinates
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        raw = d / "raw.jpg"; raw.write_bytes(img_bytes)
        with Image.open(raw) as im:
            im = im.resize((768, 1024), Image.Resampling.LANCZOS)
            im.save(d / "resized.jpg", "JPEG", quality=95)

        bindi_mask = [(364,185,404,215)]
        sindoor_mask = [(375,140,395,190)]
        nose_mask = [(315,255,335,270)]
        necklace_mask = [(300,480,470,560)]

        img_bytes = (d / "resized.jpg").read_bytes()
        img_bytes = inpaint_region(img_bytes, bindi_mask, INPAINT_PROMPTS["bindi"])
        img_bytes = inpaint_region(img_bytes, sindoor_mask, INPAINT_PROMPTS["sindoor"])
        img_bytes = inpaint_region(img_bytes, nose_mask, INPAINT_PROMPTS["nose_stud"])
        img_bytes = inpaint_region(img_bytes, necklace_mask, INPAINT_PROMPTS["necklace"])
        return img_bytes

# Watermark removal – same inpainting
def remove_watermark(img_bytes):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img_path = d / "wm.jpg"; img_path.write_bytes(img_bytes)
        with Image.open(img_path) as im:
            w, h = im.size
            mask = Image.new("L", im.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse([int(w*0.7), int(h*0.45), int(w*0.9), int(h*0.6)], fill=255)
            mask_path = d / "mask.png"; mask.save(mask_path)
        return run_model(
            INPAINT_MODEL,
            img_path,
            {
                "mask": open(mask_path, "rb"),
                "prompt": "clean skin, no text, no logo, smooth",
                "negative_prompt": NEG_PROMPT,
                "num_inference_steps": 20,
            }
        )

# Background removal – briaai/rmbg-1.4 (public)
def remove_bg(img_bytes):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inp = d / "bg_inp.jpg"
        make_gpu_safe(io.BytesIO(img_bytes), inp)  # no gpu_safe function yet, we'll use the global one
        gpu_safe_input = d / "gpu_safe.jpg"
        make_gpu_safe(inp, gpu_safe_input)  # correct call
        transparent = run_model("briaai/rmbg-1.4", gpu_safe_input, {}, image_field="image")
        return transparent

# Custom make_gpu_safe already defined above; we'll use the global gpu_safe function.
# But the remove_bg above used make_gpu_safe; we should adjust the call to use the global gpu_safe.
# Let's refactor remove_bg to directly use the bytes.
def remove_bg_fixed(img_bytes):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inp = d / "bg_inp.jpg"
        inp.write_bytes(img_bytes)
        gpu_safe_path = d / "gpu_safe.jpg"
        gpu_safe(inp, gpu_safe_path)
        transparent = run_model("briaai/rmbg-1.4", gpu_safe_path, {}, image_field="image")
        return transparent


# ---------- FULL PIPELINE ----------
def enhance_and_generate(original_bytes):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        raw = d / "raw.jpg"; raw.write_bytes(original_bytes)

        # 1. Face restore
        restored = restore_face(raw)
        (d / "restored.png").write_bytes(restored)

        # 2. Upscale
        upscaled = upscale(d / "restored.png")
        (d / "upscaled.png").write_bytes(upscaled)

        # 3. Style transfer
        styled = style_transfer(d / "upscaled.png")
        (d / "styled.png").write_bytes(styled)

        # 4. KYC markers
        marked = add_markers((d / "styled.png").read_bytes())
        (d / "marked.png").write_bytes(marked)

        # 5. Watermark removal
        clean = remove_watermark((d / "marked.png").read_bytes())
        (d / "clean.png").write_bytes(clean)

        # 6. Background removal
        no_bg = remove_bg_fixed((d / "clean.png").read_bytes())

        # Composite on grey canvas
        with Image.open(io.BytesIO(no_bg)) as sub:
            sub = ImageOps.exif_transpose(sub).convert("RGBA")
            alpha = sub.getchannel("A").filter(ImageFilter.GaussianBlur(0.9))
            sub.putalpha(alpha)
            sub.thumbnail((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), GREY_RGB + (255,))
            left = (OUTPUT_WIDTH - sub.width)//2
            top = (OUTPUT_HEIGHT - sub.height)//2
            canvas.alpha_composite(sub, (left, top))
            final = canvas.convert("RGB")
            final = ImageEnhance.Sharpness(final).enhance(1.06)

        out = io.BytesIO()
        final.save(out, format="JPEG", quality=100, subsampling=0, optimize=True)
        return out.getvalue(), final.size


# ---------- TELEGRAM HANDLERS (unchanged) ----------
async def start(update, context): ...        # same
async def recv_photo(update, context): ...   # same
async def done(update, context): ...         # same
async def cancel(update, context): ...       # same
async def status_cmd(update, context): ...   # same
async def err(update, context): ...          # same

def main():
    if not BOT_TOKEN or not REPLICATE_API_TOKEN:
        raise RuntimeError("Missing env vars")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("done", done))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, recv_photo))
    app.add_error_handler(err)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
