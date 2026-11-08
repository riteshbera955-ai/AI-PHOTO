import os
import io
import asyncio
import logging
import tempfile
import time
import threading
from pathlib import Path

import replicate
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_BATCH_PHOTOS = int(os.getenv("MAX_BATCH_PHOTOS", "20"))
OUTPUT_LONG_EDGE = int(os.getenv("OUTPUT_LONG_EDGE", "3840"))
GREY_RGB = (126, 126, 126)  # #7E7E7E

# Replicate account limit in the user's logs: 6 prediction creations/minute.
# Two models are called per photo, so keep at least 12 seconds between
# prediction creation requests and retry throttled requests.
MIN_SECONDS_BETWEEN_PREDICTIONS = 15
MAX_RATE_LIMIT_RETRIES = 6
_request_lock = threading.Lock()
_last_prediction_started = 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("brothers-photo-bot")

if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


def output_to_bytes(output):
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    if hasattr(output, "read"):
        return output.read()
    if isinstance(output, (list, tuple)) and output:
        return output_to_bytes(output[0])
    if isinstance(output, str) and output.startswith("http"):
        import requests
        response = requests.get(output, timeout=180)
        response.raise_for_status()
        return response.content
    raise RuntimeError(
        f"Unexpected Replicate output type: {type(output).__name__}"
    )


def is_rate_limit_error(exc):
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status == 429 or "429" in message or "throttl" in message or "rate limit" in message


def _retry_after_seconds(exc, fallback):
    """Read Retry-After when the SDK exposes response headers."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return max(float(value), 1.0) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def run_replicate_model(model_name, image_path, input_data):
    """Serialize Replicate calls and retry throttling without losing a batch."""
    global _last_prediction_started

    with _request_lock:
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            elapsed = time.monotonic() - _last_prediction_started
            wait_for_gap = MIN_SECONDS_BETWEEN_PREDICTIONS - elapsed
            if wait_for_gap > 0:
                log.info("Replicate request spacing: waiting %.1f seconds", wait_for_gap)
                time.sleep(wait_for_gap)

            try:
                with open(image_path, "rb") as source:
                    payload = dict(input_data)
                    payload["image"] = source
                    _last_prediction_started = time.monotonic()
                    result = replicate.run(model_name, input=payload)
                return output_to_bytes(result)

            except Exception as exc:
                if not is_rate_limit_error(exc) or attempt >= MAX_RATE_LIMIT_RETRIES:
                    raise

                fallback = min(20 * (2 ** attempt), 180)
                delay = _retry_after_seconds(exc, fallback)
                log.warning(
                    "Replicate rate limit for %s; retry %s/%s in %.1f seconds",
                    model_name, attempt + 1, MAX_RATE_LIMIT_RETRIES, delay
                )
                time.sleep(delay)

    raise RuntimeError("Replicate request could not be completed after retries.")


def enhance_and_remove_background(original_bytes):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        source_path = temp_dir / "input_image"
        source_path.write_bytes(original_bytes)

        # Model 1: 4x enhancement. Face restoration is deliberately disabled
        # to reduce identity drift and avoid inventing facial features.
        upscaled = run_replicate_model(
            "nightmareai/real-esrgan",
            source_path,
            {"scale": 4, "face_enhance": False}
        )

        upscale_path = temp_dir / "upscaled.png"
        upscale_path.write_bytes(upscaled)

        # Model 2: remove original background and preserve alpha.
        transparent = run_replicate_model(
            "bria/remove-background",
            upscale_path,
            {"preserve_alpha": True, "content_moderation": False}
        )

        with Image.open(io.BytesIO(transparent)) as im:
            im.load()
            subject = ImageOps.exif_transpose(im).convert("RGBA")

        # Gently soften the alpha edge to reduce harsh cutout outlines around
        # hair and shoulders without cropping or changing the subject shape.
        alpha = subject.getchannel("A").filter(ImageFilter.GaussianBlur(radius=0.55))
        subject.putalpha(alpha)

        width, height = subject.size
        factor = OUTPUT_LONG_EDGE / max(width, height)
        new_size = (
            max(1, round(width * factor)),
            max(1, round(height * factor))
        )
        if new_size != subject.size:
            subject = subject.resize(new_size, Image.Resampling.LANCZOS)

        canvas = Image.new("RGBA", new_size, GREY_RGB + (255,))
        canvas.alpha_composite(subject)
        final_image = canvas.convert("RGB")
        # Mild contrast and sharpening retain texture without over-smoothing skin.
        final_image = ImageEnhance.Contrast(final_image).enhance(1.03)
        final_image = ImageEnhance.Sharpness(final_image).enhance(1.15)

        output = io.BytesIO()
        final_image.save(
            output, format="JPEG", quality=96,
            subsampling=0, optimize=True
        )
        return output.getvalue(), final_image.size


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["batch"] = []
    await update.effective_message.reply_text(
        "👋 BROTHERS AI PHOTO ENHANCER\n\n"
        "Send photos one by one or as an album, then send /done.\n"
        "Output: natural face detail, grey background and up to 4K long-edge resolution.\n"
        "Processing is sequential with automatic waiting/retry for API rate limits.\n\n"
        "/done — process queued photos\n"
        "/cancel — clear the queue\n"
        "/status — check bot status"
    )


async def receive_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    media = msg.photo[-1] if msg.photo else msg.document

    if not media:
        return
    if msg.document and msg.document.mime_type and not msg.document.mime_type.startswith("image/"):
        await msg.reply_text("Please send an image file (JPG, PNG, or WEBP).")
        return
    if media.file_size and media.file_size > MAX_UPLOAD_MB * 1024 * 1024:
        await msg.reply_text(f"Image is too large. Maximum upload: {MAX_UPLOAD_MB} MB.")
        return

    batch = context.user_data.setdefault("batch", [])
    if len(batch) >= MAX_BATCH_PHOTOS:
        await msg.reply_text(
            f"Batch limit reached ({MAX_BATCH_PHOTOS}). Send /done to process it."
        )
        return

    batch.append(media.file_id)
    await msg.reply_text(
        f"📥 Added to batch: {len(batch)}/{MAX_BATCH_PHOTOS}.\n"
        "Send more photos or type /done."
    )


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    batch = context.user_data.get("batch", [])

    if not batch:
        await msg.reply_text("No photos queued. Send photos first.")
        return
    if not REPLICATE_API_TOKEN:
        await msg.reply_text(
            "❌ Server setup error: REPLICATE_API_TOKEN is missing in Render Environment."
        )
        return

    context.user_data["batch"] = []
    status_msg = await msg.reply_text(
        f"⏳ Starting {len(batch)} photo(s).\n"
        "The bot will wait automatically if Replicate rate-limits a request."
    )
    success = 0
    failed = 0

    for index, file_id in enumerate(batch, start=1):
        try:
            await status_msg.edit_text(
                f"⏳ Processing photo {index}/{len(batch)}\n"
                f"✅ Completed: {success} | ❌ Failed: {failed}\n"
                "API requests are spaced out; throttled requests retry automatically."
            )

            tg_file = await context.bot.get_file(file_id)
            raw = bytes(await tg_file.download_as_bytearray())

            result_bytes, dimensions = await asyncio.to_thread(
                enhance_and_remove_background, raw
            )

            await msg.reply_document(
                document=io.BytesIO(result_bytes),
                filename=f"brothers_4k_grey_{index}.jpg",
                caption=(
                    f"✅ Enhanced photo {index}/{len(batch)}\n"
                    f"Output: {dimensions[0]} × {dimensions[1]} px\n"
                    "Background: Grey (#7E7E7E)\n"
                    "Full frame retained; no intentional crop or stretch."
                )
            )
            success += 1

        except Exception as exc:
            failed += 1
            log.exception("Photo %s processing failed", index)
            error_text = str(exc)
            if REPLICATE_API_TOKEN:
                error_text = error_text.replace(REPLICATE_API_TOKEN, "[hidden token]")
            await msg.reply_text(
                f"❌ Photo {index} failed.\n"
                f"Error: {error_text[:450]}\n"
                "If this is an API rate/credit limit, the bot has retried automatically. "
                "Check Replicate Billing/Usage and Render Logs."
            )

    await status_msg.edit_text(
        f"🏁 Batch finished!\n"
        f"✅ Successful: {success}\n"
        f"❌ Failed: {failed}\n\n"
        "Send another batch whenever you are ready."
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    count = len(context.user_data.get("batch", []))
    context.user_data["batch"] = []
    await update.effective_message.reply_text(f"🗑️ Cleared {count} queued photo(s).")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("🟢 Bot is running.")


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
