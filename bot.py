import os
import io
import asyncio
import logging
from PIL import Image, ImageOps
import replicate
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

BOT_TOKEN = os.getenv("BOT_TOKEN")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_BATCH_PHOTOS = int(os.getenv("MAX_BATCH_PHOTOS", "30"))

# Fixed output profile based on the sample photo the user provided.
OUTPUT_WIDTH = 1030
OUTPUT_HEIGHT = 1536
TARGET_FILE_SIZE_KB = 286

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

WELCOME = (
    "👋 Welcome to BROTHERS AI Photo Enhancer!\\n\\n"
    "Send multiple photos one by one or as an album. "
    "When finished, send /done to enhance the whole batch.\\n\\n"
    "Every output is exported as JPG at 1030 × 1536 px. "
    "File size is targeted near 286 KB (it can vary with image detail).\\n"
    "Use /cancel to clear the current batch."
)

def get_image_bytes(result):
    if hasattr(result, "read"):
        return result.read()
    if isinstance(result, (list, tuple)) and result:
        item = result[0]
        if hasattr(item, "read"):
            return item.read()
        if isinstance(item, (bytes, bytearray)):
            return bytes(item)
    raise RuntimeError("The AI service returned an unsupported output.")

def encode_jpeg_near_target(image: Image.Image):
    """Choose the highest JPEG quality that stays near/below target size."""
    target_bytes = TARGET_FILE_SIZE_KB * 1024
    best_data = None
    best_gap = float("inf")

    # Try a range of JPEG quality values; size depends on image complexity.
    for quality in range(95, 54, -2):
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True, progressive=True)
        data = output.getvalue()
        gap = abs(len(data) - target_bytes)
        if gap < best_gap:
            best_data = data
            best_gap = gap
        if len(data) <= target_bytes:
            # First under-target result is generally the best quality under the cap.
            return data

    return best_data

def enhance_image(image_bytes: bytes):
    # Run AI enhancement remotely, then export to the requested fixed dimensions.
    result = replicate.run(
        "nightmareai/real-esrgan",
        input={
            "image": io.BytesIO(image_bytes),
            "scale": 4,
            "face_enhance": True
        }
    )
    enhanced_bytes = get_image_bytes(result)

    with Image.open(io.BytesIO(enhanced_bytes)) as enhanced:
        enhanced.load()
        enhanced = ImageOps.exif_transpose(enhanced).convert("RGB")

        # Exact 1030x1536 output. This resizes to the requested canvas;
        # aspect ratios different from the sample may appear slightly stretched.
        enhanced = enhanced.resize(
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            Image.Resampling.LANCZOS
        )
        output_bytes = encode_jpeg_near_target(enhanced)

    return output_bytes, (OUTPUT_WIDTH, OUTPUT_HEIGHT)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["batch"] = []
    await update.message.reply_text(WELCOME)

async def receive_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if not BOT_TOKEN or not REPLICATE_API_TOKEN:
        await message.reply_text("⚠️ Bot configuration is incomplete. Please contact the bot owner.")
        return

    photo = message.photo[-1] if message.photo else message.document
    if not photo:
        await message.reply_text("Please send an image.")
        return

    if photo.file_size and photo.file_size > MAX_UPLOAD_MB * 1024 * 1024:
        await message.reply_text(f"Image is too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")
        return

    batch = context.user_data.setdefault("batch", [])
    if len(batch) >= MAX_BATCH_PHOTOS:
        await message.reply_text(
            f"Batch limit is {MAX_BATCH_PHOTOS} photos. Send /done to process this batch."
        )
        return

    batch.append({"file_id": photo.file_id})
    await message.reply_text(
        f"📥 Added to batch: {len(batch)}/{MAX_BATCH_PHOTOS} photos.\\n"
        "Send more photos or type /done to start enhancement."
    )

async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    batch = context.user_data.get("batch", [])
    if not batch:
        await message.reply_text("Your batch is empty. Send some photos first.")
        return

    context.user_data["batch"] = []
    status = await message.reply_text(f"🚀 Starting AI enhancement for {len(batch)} photos...")
    success, failed = 0, 0

    for index, item in enumerate(batch, start=1):
        try:
            await status.edit_text(
                f"⏳ Enhancing photo {index}/{len(batch)}...\\n"
                f"Completed: {success} | Failed: {failed}"
            )
            tg_file = await context.bot.get_file(item["file_id"])
            downloaded = await tg_file.download_as_bytearray()
            enhanced, dimensions = await asyncio.to_thread(
                enhance_image, bytes(downloaded)
            )
            await message.reply_document(
                document=io.BytesIO(enhanced),
                filename=f"enhanced_{index}.jpg",
                caption=(
                    f"✅ Enhanced photo {index}/{len(batch)}\\n"
                    f"Output dimensions: {dimensions[0]} × {dimensions[1]} px\\n"
                    f"File size: {len(enhanced) / 1024:.1f} KB"
                )
            )
            success += 1
        except Exception as exc:
            failed += 1
            log.exception("Enhancement failed for item %s: %s", index, exc)

    await status.edit_text(
        f"🏁 Batch finished!\\n"
        f"✅ Successful: {success}\\n"
        f"❌ Failed: {failed}\\n\\n"
        "Send another batch whenever you are ready."
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    count = len(context.user_data.get("batch", []))
    context.user_data["batch"] = []
    await update.message.reply_text(f"🗑️ Cleared {count} queued photo(s).")

async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🟢 Bot is running.")

def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN environment variable is missing.")
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN environment variable is missing.")

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("done", done))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, receive_photo))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
