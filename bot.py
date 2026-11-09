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
from PIL import Image, ImageOps
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()

MAX_UPLOAD_MB = 15
MAX_BATCH_PHOTOS = 20
OUTPUT_WIDTH = 1030
OUTPUT_HEIGHT = 1536

BRIA_MODEL = "bria/increase-resolution"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

log = logging.getLogger("BROTHERS_PHOTO_BOT")

_call_lock = threading.Lock()
_last_call = 0.0

if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


def output_to_bytes(output):

    if isinstance(output, (bytes, bytearray)):
        return bytes(output)

    if hasattr(output, "read"):
        data = output.read()
        if data:
            return data

    if isinstance(output, (list, tuple)) and output:
        return output_to_bytes(output[0])

    if isinstance(output, str) and output.startswith("http"):
        response = requests.get(output, timeout=180)
        response.raise_for_status()
        return response.content

    url = getattr(output, "url", None)

    if callable(url):
        url = url()

    if isinstance(url, str) and url.startswith("http"):
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        return response.content

    raise RuntimeError("Unexpected Replicate output")


def run_replicate(model, inputs):

    global _last_call

    with _call_lock:

        gap = 12 - (time.monotonic() - _last_call)

        if gap > 0:
            time.sleep(gap)

        _last_call = time.monotonic()

        result = replicate.run(
            model,
            input=inputs
        )

        return output_to_bytes(result)


def passport_crop(image):

    image = ImageOps.exif_transpose(image).convert("RGB")

    width = 413
    height = 531

    target_ratio = width / height
    ratio = image.width / image.height

    if ratio > target_ratio:

        new_width = int(image.height * target_ratio)
        left = (image.width - new_width) // 2

        image = image.crop((
            left,
            0,
            left + new_width,
            image.height
        ))

    else:

        new_height = int(image.width / target_ratio)
        top = max(0, (image.height - new_height) // 2)

        image = image.crop((
            0,
            top,
            image.width,
            top + new_height
        ))

    return image.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )


def enhance_photo(original_bytes, passport_mode=False):

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        input_path = temp_dir / "input.jpg"
        normalized_path = temp_dir / "normalized.jpg"

        input_path.write_bytes(original_bytes)

        with Image.open(input_path) as image:

            original = ImageOps.exif_transpose(image).convert("RGB")

            original.save(
                normalized_path,
                "JPEG",
                quality=96
            )

        # Bria enhancement only.
        # No GFPGAN or generative face reconstruction.

        with open(normalized_path, "rb") as image_file:

            enhanced_bytes = run_replicate(
                BRIA_MODEL,
                {
                    "image": image_file,
                    "desired_increase": 2,
                    "preserve_alpha": False,
                    "sync": True,
                    "content_moderation": False
                }
            )

        with Image.open(io.BytesIO(enhanced_bytes)) as image:

            final = ImageOps.exif_transpose(image).convert("RGB")

        if passport_mode:

            final = passport_crop(final)

        else:

            final = final.resize(
                (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                Image.Resampling.LANCZOS
            )

        output = io.BytesIO()

        final.save(
            output,
            "JPEG",
            quality=95,
            subsampling=0,
            optimize=True
        )

        return output.getvalue(), final.size


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    context.user_data["batch"] = []
    context.user_data["passport_mode"] = False

    await update.message.reply_text(
        "👋 BROTHERS PHOTO RESTORER\n\n"
        "Send photos and then /done.\n\n"
        "✨ Bria Image Enhancement\n"
        "✨ No generative face reconstruction\n"
        "✨ Original background retained\n\n"
        "Normal: 1030 × 1536 px\n"
        "Passport: 413 × 531 px\n\n"
        "/passport - Passport mode\n"
        "/normal - Normal mode\n"
        "/done - Process photos\n"
        "/cancel - Clear queue\n"
        "/status - Bot status"
    )


async def passport(update: Update, context: ContextTypes.DEFAULT_TYPE):

    context.user_data["passport_mode"] = True

    await update.message.reply_text(
        "🪪 Passport mode enabled.\n"
        "Output: 413 × 531 px"
    )


async def normal(update: Update, context: ContextTypes.DEFAULT_TYPE):

    context.user_data["passport_mode"] = False

    await update.message.reply_text(
        "✅ Normal mode enabled.\n"
        "Output: 1030 × 1536 px"
    )


async def receive_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    message = update.effective_message

    media = (
        message.photo[-1]
        if message.photo
        else message.document
    )

    if not media:
        return

    if message.document and message.document.mime_type:
        if not message.document.mime_type.startswith("image/"):
            await message.reply_text("Send JPG, PNG or WEBP images.")
            return

    if media.file_size and media.file_size > MAX_UPLOAD_MB * 1024 * 1024:
        await message.reply_text("Maximum image size is 15 MB.")
        return

    batch = context.user_data.setdefault("batch", [])

    if len(batch) >= MAX_BATCH_PHOTOS:
        await message.reply_text("Batch limit reached. Send /done.")
        return

    batch.append(media.file_id)

    await message.reply_text(
        f"📥 Added: {len(batch)}/{MAX_BATCH_PHOTOS}\n"
        "Send more photos or /done."
    )


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):

    message = update.effective_message

    batch = context.user_data.get("batch", [])

    if not batch:
        await message.reply_text("No photos queued.")
        return

    if not REPLICATE_API_TOKEN:
        await message.reply_text("REPLICATE_API_TOKEN missing.")
        return

    active = context.application.bot_data.setdefault(
        "processing_users",
        set()
    )

    user_id = update.effective_user.id

    if user_id in active:
        await message.reply_text("Your previous batch is processing.")
        return

    active.add(user_id)

    context.user_data["batch"] = []

    passport_mode = context.user_data.get(
        "passport_mode",
        False
    )

    status_message = await message.reply_text(
        f"⏳ Processing {len(batch)} photos..."
    )

    success = 0
    failed = 0

    try:

        for index, file_id in enumerate(batch, 1):

            try:

                await status_message.edit_text(
                    f"⏳ Processing {index}/{len(batch)}\n"
                    f"✅ Completed: {success}\n"
                    f"❌ Failed: {failed}"
                )

                telegram_file = await context.bot.get_file(file_id)

                raw = bytes(
                    await telegram_file.download_as_bytearray()
                )

                result, dimensions = await asyncio.to_thread(
                    enhance_photo,
                    raw,
                    passport_mode
                )

                await message.reply_document(
                    document=io.BytesIO(result),
                    filename=f"brothers_restored_{index}.jpg",
                    caption=(
                        f"✅ Photo {index} completed\n"
                        f"Size: {dimensions[0]} × {dimensions[1]} px\n"
                        "Model: Bria Enhancement"
                    )
                )

                success += 1

            except Exception as error:

                failed += 1

                log.exception("Photo processing failed")

                error_text = str(error).replace(
                    REPLICATE_API_TOKEN,
                    "[hidden]"
                )

                await message.reply_text(
                    f"❌ Photo {index} failed:\n{error_text[:400]}"
                )

    finally:

        active.discard(user_id)

    await status_message.edit_text(
        "🏁 Batch completed!\n"
        f"✅ Successful: {success}\n"
        f"❌ Failed: {failed}"
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):

    count = len(context.user_data.get("batch", []))

    context.user_data["batch"] = []

    await update.message.reply_text(
        f"🗑️ Cleared {count} queued photos."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):

    active = context.application.bot_data.setdefault(
        "processing_users",
        set()
    )

    user_id = update.effective_user.id

    await update.message.reply_text(
        "🟢 BROTHERS PHOTO BOT ONLINE\n\n"
        f"Queued: {len(context.user_data.get('batch', []))}\n"
        f"Processing: {'Yes' if user_id in active else 'No'}\n"
        f"Passport mode: {'ON' if context.user_data.get('passport_mode') else 'OFF'}"
    )


async def error_handler(update, context):

    log.exception(
        "Telegram error",
        exc_info=context.error
    )


def main():

    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is missing.")

    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN is missing.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("passport", passport))
    app.add_handler(CommandHandler("normal", normal))
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
