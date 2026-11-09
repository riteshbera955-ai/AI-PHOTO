import os
import io
import asyncio
import logging
import tempfile
import time
import threading
from pathlib import Path

import replicate
from PIL import Image, ImageOps, ImageEnhance
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()

MAX_UPLOAD_MB = 15
MAX_BATCH_PHOTOS = 20
OUTPUT_WIDTH = 1030
OUTPUT_HEIGHT = 1536

CODEFORMER_FIDELITY = 0.98

MIN_SECONDS_BETWEEN_PREDICTIONS = 15
MAX_RATE_LIMIT_RETRIES = 6

_request_lock = threading.Lock()
_last_prediction_started = 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

log = logging.getLogger("brothers-photo-bot")

if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


def make_safe_image(input_path, output_path):
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")

        max_pixels = 1800000
        width, height = image.size

        if width * height > max_pixels:
            scale = (max_pixels / (width * height)) ** 0.5
            image = image.resize(
                (int(width * scale), int(height * scale)),
                Image.Resampling.LANCZOS
            )

        image.save(output_path, "JPEG", quality=95)

    return output_path


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

    raise RuntimeError("Unexpected Replicate output")


def is_rate_limit_error(exc):
    message = str(exc).lower()
    status = getattr(exc, "status", None)

    return (
        status == 429
        or "429" in message
        or "throttl" in message
        or "rate limit" in message
    )


def run_model(model_name, image_path, input_data):
    global _last_prediction_started

    with _request_lock:

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):

            elapsed = time.monotonic() - _last_prediction_started
            wait = MIN_SECONDS_BETWEEN_PREDICTIONS - elapsed

            if wait > 0:
                time.sleep(wait)

            try:
                with open(image_path, "rb") as source:

                    payload = dict(input_data)
                    payload["image"] = source

                    _last_prediction_started = time.monotonic()

                    result = replicate.run(
                        model_name,
                        input=payload
                    )

                return output_to_bytes(result)

            except Exception as exc:

                if not is_rate_limit_error(exc):
                    raise

                if attempt >= MAX_RATE_LIMIT_RETRIES:
                    raise

                delay = min(20 * (2 ** attempt), 180)

                log.warning(
                    "Rate limited. Retrying in %s seconds",
                    delay
                )

                time.sleep(delay)

    raise RuntimeError("Replicate processing failed")


def enhance_photo(original_bytes):

    with tempfile.TemporaryDirectory() as temp:

        temp = Path(temp)

        source = temp / "input.jpg"
        source.write_bytes(original_bytes)

        safe_image = temp / "safe.jpg"

        make_safe_image(source, safe_image)

        # FACE RESTORATION
        restored = run_model(
            "sczhou/codeformer:cc4956dd26fa5a7185d5660cc9100fab1b8070a1d1654a8bb5eb6d443b020bb2",
            safe_image,
            {
                "codeformer_fidelity": CODEFORMER_FIDELITY,
                "background_enhance": True,
                "face_upsample": True,
                "upscale": 2
            }
        )

        restored_path = temp / "restored.png"
        restored_path.write_bytes(restored)

        safe_restored = temp / "restored_safe.jpg"
        make_safe_image(restored_path, safe_restored)

        # IMAGE UPSCALING
        upscaled = run_model(
            "nightmareai/real-esrgan",
            safe_restored,
            {
                "scale": 4,
                "face_enhance": False
            }
        )

        with Image.open(io.BytesIO(upscaled)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")

        image.thumbnail(
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "RGB",
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            image.getpixel((0, 0))
        )

        x = (OUTPUT_WIDTH - image.width) // 2
        y = (OUTPUT_HEIGHT - image.height) // 2

        canvas.paste(image, (x, y))

        final_image = ImageEnhance.Sharpness(canvas).enhance(1.03)

        output = io.BytesIO()

        final_image.save(
            output,
            format="JPEG",
            quality=96,
            subsampling=0,
            optimize=True
        )

        return output.getvalue()


async def start(update, context):

    context.user_data["batch"] = []

    await update.effective_message.reply_text(
        "BROTHERS AI PHOTO ENHANCER\n\n"
        "Send your photos and type /done.\n\n"
        "Original background retained.\n"
        "Natural photo enhancement.\n"
        "1030 x 1536 JPG output.\n\n"
        "/done - Process photos\n"
        "/cancel - Clear queue\n"
        "/status - Bot status"
    )


async def receive_photo(update, context):

    msg = update.effective_message

    media = msg.photo[-1] if msg.photo else msg.document

    if not media:
        return

    if (
        msg.document
        and msg.document.mime_type
        and not msg.document.mime_type.startswith("image/")
    ):
        await msg.reply_text("Send JPG, PNG or WEBP images only.")
        return

    if media.file_size and media.file_size > MAX_UPLOAD_MB * 1024 * 1024:
        await msg.reply_text("Maximum image size is 15 MB.")
        return

    batch = context.user_data.setdefault("batch", [])

    if len(batch) >= MAX_BATCH_PHOTOS:
        await msg.reply_text("Batch limit reached. Send /done.")
        return

    batch.append(media.file_id)

    await msg.reply_text(
        f"Photo added: {len(batch)}/{MAX_BATCH_PHOTOS}\n"
        "Send more photos or type /done."
    )


async def done(update, context):

    msg = update.effective_message
    batch = context.user_data.get("batch", [])

    if not batch:
        await msg.reply_text("No photos queued.")
        return

    if not REPLICATE_API_TOKEN:
        await msg.reply_text("Replicate API token missing.")
        return

    processing = context.application.bot_data.setdefault(
        "processing_users", set()
    )

    user_id = update.effective_user.id

    if user_id in processing:
        await msg.reply_text("Your previous batch is still processing.")
        return

    processing.add(user_id)
    context.user_data["batch"] = []

    status_msg = await msg.reply_text(
        f"Starting {len(batch)} photo(s)..."
    )

    success = 0
    failed = 0

    try:

        for index, file_id in enumerate(batch, 1):

            try:

                await status_msg.edit_text(
                    f"Processing {index}/{len(batch)}\n"
                    f"Completed: {success}\n"
                    f"Failed: {failed}"
                )

                tg_file = await context.bot.get_file(file_id)

                raw = bytes(
                    await tg_file.download_as_bytearray()
                )

                result = await asyncio.to_thread(
                    enhance_photo,
                    raw
                )

                await msg.reply_document(
                    document=io.BytesIO(result),
                    filename=f"brothers_enhanced_{index}.jpg",
                    caption=(
                        f"Enhanced photo {index}/{len(batch)}\n"
                        "Original background retained.\n"
                        "Output: 1030 x 1536 JPG"
                    )
                )

                success += 1

            except Exception as exc:

                failed += 1

                log.exception("Photo processing failed")

                error = str(exc).replace(
                    REPLICATE_API_TOKEN,
                    "[hidden]"
                )

                await msg.reply_text(
                    f"Photo {index} failed:\n{error[:400]}"
                )

    finally:
        processing.discard(user_id)

    await status_msg.edit_text(
        "Batch finished!\n\n"
        f"Successful: {success}\n"
        f"Failed: {failed}"
    )


async def cancel(update, context):

    count = len(context.user_data.get("batch", []))
    context.user_data["batch"] = []

    await update.effective_message.reply_text(
        f"Cleared {count} queued photos."
    )


async def status(update, context):

    processing = context.application.bot_data.setdefault(
        "processing_users", set()
    )

    active = update.effective_user.id in processing
    queued = len(context.user_data.get("batch", []))

    await update.effective_message.reply_text(
        "Bot is running.\n"
        f"Queued photos: {queued}\n"
        f"Processing: {'Yes' if active else 'No'}"
    )


async def error_handler(update, context):
    log.exception(
        "Telegram handler error",
        exc_info=context.error
    )


def main():

    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN missing.")

    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN missing.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("done", done))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CommandHandler("status", status))

    app.add_handler(
        MessageHandler(
            filters.PHOTO | filters.Document.IMAGE,
            receive_photo
        )
    )

    app.add_error_handler(error_handler)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
