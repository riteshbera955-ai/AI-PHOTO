import os
import io
import asyncio
import logging
import tempfile
import time
import threading
from pathlib import Path

import replicate
import requests
from PIL import Image, ImageOps, ImageEnhance
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_BATCH_PHOTOS = int(os.getenv("MAX_BATCH_PHOTOS", "20"))
OUTPUT_WIDTH = int(os.getenv("OUTPUT_WIDTH", "1030"))
OUTPUT_HEIGHT = int(os.getenv("OUTPUT_HEIGHT", "1536"))

# Pinned, documented Replicate model versions.
SWINIR_MODEL = "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a"
GFPGAN_MODEL = "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c"

# Replicate account previously reported a low prediction-creation limit.
MIN_SECONDS_BETWEEN_PREDICTIONS = 15
MAX_RATE_LIMIT_RETRIES = 5
GPU_MAX_PIXELS = int(os.getenv("GPU_MAX_PIXELS", "1800000"))
_request_lock = threading.Lock()
_last_prediction_started = 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("brothers-photo-bot")

if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


def make_safe_image(input_path, output_path):
    """Normalize orientation and keep the complete image within GPU pixel limits."""
    with Image.open(input_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        if w * h > GPU_MAX_PIXELS:
            factor = (GPU_MAX_PIXELS / float(w * h)) ** 0.5
            im = im.resize(
                (max(1, int(w * factor)), max(1, int(h * factor))),
                Image.Resampling.LANCZOS
            )
        im.save(output_path, "JPEG", quality=95, optimize=True)
    return output_path


def output_to_bytes(output):
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    if hasattr(output, "read"):
        data = output.read()
        if data:
            return data
    if isinstance(output, (list, tuple)) and output:
        return output_to_bytes(output[0])
    if isinstance(output, dict):
        # Some Replicate wrappers return a file URL in a dictionary.
        for key in ("url", "file", "output"):
            if output.get(key):
                return output_to_bytes(output[key])
    if isinstance(output, str) and output.startswith("http"):
        response = requests.get(output, timeout=180)
        response.raise_for_status()
        return response.content
    # Newer Replicate FileOutput objects can expose a URL property.
    url = getattr(output, "url", None)
    if callable(url):
        url = url()
    if isinstance(url, str) and url.startswith("http"):
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        return response.content
    raise RuntimeError(f"Unexpected Replicate output type: {type(output).__name__}")


def is_rate_limit_error(exc):
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status == 429 or "429" in message or "throttl" in message or "rate limit" in message


def retry_after_seconds(exc, fallback):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return max(float(value), 1.0) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def run_model(model_name, image_path, input_data, image_field="image"):
    """Run one Replicate prediction at a time, retrying rate limits."""
    global _last_prediction_started
    with _request_lock:
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            gap = MIN_SECONDS_BETWEEN_PREDICTIONS - (time.monotonic() - _last_prediction_started)
            if gap > 0:
                log.info("Replicate request spacing: waiting %.1f seconds", gap)
                time.sleep(gap)
            try:
                with open(image_path, "rb") as source:
                    payload = dict(input_data)
                    payload[image_field] = source
                    _last_prediction_started = time.monotonic()
                    result = replicate.run(model_name, input=payload)
                return output_to_bytes(result)
            except Exception as exc:
                if not is_rate_limit_error(exc) or attempt >= MAX_RATE_LIMIT_RETRIES:
                    raise
                delay = retry_after_seconds(exc, min(20 * (2 ** attempt), 180))
                log.warning("Rate limit on %s; retry %s/%s in %.1fs", model_name, attempt + 1, MAX_RATE_LIMIT_RETRIES, delay)
                time.sleep(delay)
    raise RuntimeError("Replicate request failed after retries")


def fit_exact_canvas(image, width, height):
    """Fit without cropping or stretching; use an edge-colour canvas for any small margins."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    # Edge-colour padding avoids cropping the original scene or distorting people.
    px = image.load()
    w, h = image.size
    samples = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1]]
    background = tuple(sum(c[i] for c in samples) // len(samples) for i in range(3))
    canvas = Image.new("RGB", (width, height), background)
    canvas.paste(image, ((width - w) // 2, (height - h) // 2))
    return canvas


def enhance_photo(original_bytes):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source_path = tmp / "input.jpg"
        source_path.write_bytes(original_bytes)
        safe_path = tmp / "input_safe.jpg"
        make_safe_image(source_path, safe_path)

        # Stage 1: full-frame restoration/upscaling. Original background remains in-frame.
        swinir_bytes = run_model(
            SWINIR_MODEL,
            safe_path,
            {
                "task_type": "Real-World Image Super-Resolution",
                "noise": 15,
                "jpeg": 40,
            },
            image_field="image",
        )
        swinir_path = tmp / "swinir.png"
        swinir_path.write_bytes(swinir_bytes)
        swinir_safe = tmp / "swinir_safe.jpg"
        make_safe_image(swinir_path, swinir_safe)

        # Stage 2: face-only restoration pass; GFPGAN keeps the full scene in its output.
        gfpgan_bytes = run_model(
            GFPGAN_MODEL,
            swinir_safe,
            {"version": "v1.4", "scale": 2},
            image_field="img",
        )

        with Image.open(io.BytesIO(gfpgan_bytes)) as im:
            im.load()
            final_image = fit_exact_canvas(im, OUTPUT_WIDTH, OUTPUT_HEIGHT)

        # Very mild sharpening only; do not add synthetic detail aggressively.
        final_image = ImageEnhance.Sharpness(final_image).enhance(1.02)
        out = io.BytesIO()
        final_image.save(out, format="JPEG", quality=96, subsampling=0, optimize=True)
        return out.getvalue(), final_image.size


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["batch"] = []
    await update.effective_message.reply_text(
        "👋 BROTHERS AI PHOTO RESTORER\n\n"
        "Send photos, then /done to process.\n"
        "Models: SwinIR (full-frame restoration) + GFPGAN (face restoration).\n"
        "Original scene/background is retained; no background removal.\n"
        f"Output: {OUTPUT_WIDTH} × {OUTPUT_HEIGHT} JPG.\n\n"
        "/done — process queued photos\n/cancel — clear queue\n/status — check status"
    )


async def receive_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    media = msg.photo[-1] if msg.photo else msg.document
    if not media:
        return
    if msg.document and msg.document.mime_type and not msg.document.mime_type.startswith("image/"):
        await msg.reply_text("Please send a JPG, PNG, or WEBP image.")
        return
    if media.file_size and media.file_size > MAX_UPLOAD_MB * 1024 * 1024:
        await msg.reply_text(f"Image too large. Maximum: {MAX_UPLOAD_MB} MB.")
        return
    batch = context.user_data.setdefault("batch", [])
    if len(batch) >= MAX_BATCH_PHOTOS:
        await msg.reply_text(f"Batch limit reached ({MAX_BATCH_PHOTOS}). Send /done.")
        return
    batch.append(media.file_id)
    await msg.reply_text(f"📥 Added: {len(batch)}/{MAX_BATCH_PHOTOS}. Send more or /done.")


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    batch = context.user_data.get("batch", [])
    if not batch:
        await msg.reply_text("No photos queued. Send photos first.")
        return
    if not REPLICATE_API_TOKEN:
        await msg.reply_text("❌ REPLICATE_API_TOKEN is missing in Render Environment.")
        return

    processing_users = context.application.bot_data.setdefault("processing_users", set())
    user_id = update.effective_user.id
    if user_id in processing_users:
        await msg.reply_text("⏳ Your previous batch is still processing. Please wait.")
        return
    processing_users.add(user_id)
    context.user_data["batch"] = []
    status_msg = await msg.reply_text(f"⏳ Starting {len(batch)} photo(s) with SwinIR + GFPGAN...")
    success = 0
    failed = 0
    try:
        for index, file_id in enumerate(batch, start=1):
            try:
                await status_msg.edit_text(
                    f"⏳ Processing photo {index}/{len(batch)}\n"
                    f"✅ Completed: {success} | ❌ Failed: {failed}\n"
                    "Stage 1/2: full-image restoration; Stage 2/2: face restoration."
                )
                tg_file = await context.bot.get_file(file_id)
                raw = bytes(await tg_file.download_as_bytearray())
                result, dimensions = await asyncio.to_thread(enhance_photo, raw)
                await msg.reply_document(
                    document=io.BytesIO(result),
                    filename=f"brothers_restored_{index}.jpg",
                    caption=(
                        f"✅ Restored photo {index}/{len(batch)}\n"
                        f"Output: {dimensions[0]} × {dimensions[1]} px\n"
                        "Models: SwinIR + GFPGAN v1.4\n"
                        "Original background retained; no background removal."
                    ),
                )
                success += 1
            except Exception as exc:
                failed += 1
                log.exception("Photo %s processing failed", index)
                error_text = str(exc).replace(REPLICATE_API_TOKEN, "[hidden token]")
                await msg.reply_text(f"❌ Photo {index} failed.\nError: {error_text[:450]}")
    finally:
        processing_users.discard(user_id)
    await status_msg.edit_text(
        f"🏁 Batch finished!\n✅ Successful: {success}\n❌ Failed: {failed}\n\nSend another batch whenever ready."
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    count = len(context.user_data.get("batch", []))
    context.user_data["batch"] = []
    await update.effective_message.reply_text(f"🗑️ Cleared {count} queued photo(s).")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    processing_users = context.application.bot_data.setdefault("processing_users", set())
    active = update.effective_user.id in processing_users
    queued = len(context.user_data.get("batch", []))
    await update.effective_message.reply_text(
        f"🟢 Bot is running.\nQueued photos: {queued}\nProcessing: {'Yes' if active else 'No'}"
    )


async def error_handler(update, context):
    log.exception("Unhandled Telegram handler error", exc_info=context.error)


def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is missing from Render Environment.")
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN is missing from Render Environment.")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("done", done))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, receive_photo))
    app.add_error_handler(error_handler)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
