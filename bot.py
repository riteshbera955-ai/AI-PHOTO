import os
import io
import asyncio
import logging
import tempfile

import replicate

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

NAFNET_MODEL = (
    "megvii-research/nafnet:"
    "018241a6c880319404eaa2714b764313e27e11f950a7ff0a7b5b37b27b74dcf7"
)

ESRGAN_MODEL = "nightmareai/real-esrgan"

MAX_PHOTOS = 20
MAX_SIZE = 15 * 1024 * 1024

WIDTH = 1030
HEIGHT = 1536

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


def get_output_bytes(result):

    if isinstance(result, (list, tuple)):
        result = result[0]

    if hasattr(result, "read"):
        return result.read()

    if hasattr(result, "url"):
        url = result.url
        if callable(url):
            url = url()
        import requests
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        return response.content

    if isinstance(result, str):
        import requests
        response = requests.get(result, timeout=180)
        response.raise_for_status()
        return response.content

    raise RuntimeError("Invalid Replicate output")


def run_model(model, inputs):

    result = replicate.run(model, input=inputs)

    return get_output_bytes(result)


def process_image(raw, passport=False):

    with tempfile.TemporaryDirectory() as folder:

        source = os.path.join(folder, "input.jpg")

        with Image.open(io.BytesIO(raw)) as im:
            original = ImageOps.exif_transpose(im).convert("RGB")
            original.save(source, "JPEG", quality=96)

        # STEP 1: AI DEBLURRING

        with open(source, "rb") as f:
            deblurred = run_model(
                NAFNET_MODEL,
                {
                    "image": f,
                    "task_type": "Image Debluring (REDS)"
                }
            )

        deblur_path = os.path.join(folder, "deblur.jpg")

        with open(deblur_path, "wb") as f:
            f.write(deblurred)

        # STEP 2: AI UPSCALING

        with open(deblur_path, "rb") as f:
            upscaled = run_model(
                ESRGAN_MODEL,
                {
                    "image": f,
                    "scale": 2,
                    "face_enhance": False
                }
            )

        with Image.open(io.BytesIO(upscaled)) as im:
            final = ImageOps.exif_transpose(im).convert("RGB")

        # STEP 3: PRESERVE ASPECT RATIO

        target_w, target_h = (
            (413, 531) if passport else (WIDTH, HEIGHT)
        )

        ratio = min(
            target_w / final.width,
            target_h / final.height
        )

        new_w = round(final.width * ratio)
        new_h = round(final.height * ratio)

        final = final.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS
        )

        # No stretching or forced cropping.
        # Pad remaining space using edge colour.

        canvas = Image.new(
            "RGB",
            (target_w, target_h),
            final.getpixel((0, 0))
        )

        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2

        canvas.paste(final, (x, y))

        output = io.BytesIO()

        canvas.save(
            output,
            "JPEG",
            quality=96,
            subsampling=0
        )

        output.seek(0)

        return output


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    context.user_data["batch"] = []
    context.user_data["passport"] = False

    await update.message.reply_text(
        "BROTHERS AI PHOTO RESTORER\n\n"
        "Send photos and use /done.\n\n"
        "/normal - Normal mode\n"
        "/passport - Passport mode\n"
        "/done - Process photos\n"
        "/cancel - Clear queue\n"
        "/status - Bot status"
    )


async def normal(update: Update, context: ContextTypes.DEFAULT_TYPE):

    context.user_data["passport"] = False

    await update.message.reply_text(
        "Normal mode enabled: 1030 x 1536"
    )


async def passport(update: Update, context: ContextTypes.DEFAULT_TYPE):

    context.user_data["passport"] = True

    await update.message.reply_text(
        "Passport mode enabled: 413 x 531"
    )


async def receive(update: Update, context: ContextTypes.DEFAULT_TYPE):

    msg = update.effective_message

    media = msg.photo[-1] if msg.photo else msg.document

    if not media:
        return

    if msg.document and msg.document.mime_type:
        if not msg.document.mime_type.startswith("image/"):
            await msg.reply_text("Please send an image.")
            return

    if media.file_size and media.file_size > MAX_SIZE:
        await msg.reply_text("Maximum image size is 15 MB.")
        return

    batch = context.user_data.setdefault("batch", [])

    if len(batch) >= MAX_PHOTOS:
        await msg.reply_text("Maximum 20 photos. Use /done.")
        return

    batch.append(media.file_id)

    await msg.reply_text(
        f"Photo added: {len(batch)}/{MAX_PHOTOS}"
    )


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):

    msg = update.effective_message
    batch = context.user_data.get("batch", [])

    if not batch:
        await msg.reply_text("No photos queued.")
        return

    active = context.application.bot_data.setdefault(
        "active_users", set()
    )

    uid = update.effective_user.id

    if uid in active:
        await msg.reply_text("Your batch is already processing.")
        return

    active.add(uid)
    context.user_data["batch"] = []

    passport_mode = context.user_data.get("passport", False)

    status = await msg.reply_text("Starting AI restoration...")

    success = 0
    failed = 0

    try:

        for index, file_id in enumerate(batch, 1):

            try:

                tg_file = await context.bot.get_file(file_id)

                raw = bytes(
                    await tg_file.download_as_bytearray()
                )

                result = await asyncio.to_thread(
                    process_image,
                    raw,
                    passport_mode
                )

                await msg.reply_document(
                    document=result,
                    filename=f"restored_{index}.jpg",
                    caption=f"AI restoration completed: {index}"
                )

                success += 1

            except Exception as e:

                failed += 1
                log.exception("Processing error")

                await msg.reply_text(
                    f"Photo {index} failed: {str(e)[:250]}"
                )

            await status.edit_text(
                f"Processing {index}/{len(batch)}\n"
                f"Success: {success}\n"
                f"Failed: {failed}"
            )

    finally:

        active.discard(uid)

    await msg.reply_text(
        f"Batch finished!\nSuccess: {success}\nFailed: {failed}"
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):

    count = len(context.user_data.get("batch", []))
    context.user_data["batch"] = []

    await update.message.reply_text(
        f"Queue cleared: {count}"
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):

    active = context.application.bot_data.setdefault(
        "active_users", set()
    )

    uid = update.effective_user.id

    await update.message.reply_text(
        "BOT ONLINE\n"
        f"Queued: {len(context.user_data.get('batch', []))}\n"
        f"Processing: {'Yes' if uid in active else 'No'}"
    )


def main():

    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN missing")

    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN missing")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("normal", normal))
    app.add_handler(CommandHandler("passport", passport))
    app.add_handler(CommandHandler("done", done))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CommandHandler("status", status))

    app.add_handler(
        MessageHandler(
            filters.PHOTO | filters.Document.IMAGE,
            receive
        )
    )

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
