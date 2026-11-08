import os
import io
import asyncio
import logging
from PIL import Image
import replicate
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

BOT_TOKEN = os.getenv("BOT_TOKEN")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_BATCH_PHOTOS = int(os.getenv("MAX_BATCH_PHOTOS", "30"))

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

WELCOME = (
    "👋 Welcome to BROTHERS AI Photo Enhancer!\\n\\n"
    "Send multiple photos one by one or as an album. "
    "When finished, send /done to enhance the whole batch.\\n\\n"
    "The enhanced output will keep each original photo's exact pixel dimensions. "
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

def enhance_image(image_bytes: bytes):
    # AI enhancement/upscaling runs remotely. We resize the enhanced result
    # back to the exact original width and height after inference.
    with Image.open(io.BytesIO(image_bytes)) as original:
        original.load()
        original_size = original.size
        original_format = (original.format or "JPEG").upper()
        original_mode = original.mode

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
        if original_mode in ("RGBA", "LA") and original_format == "PNG":
            enhanced = enhanced.convert("RGBA")
        else:
            enhanced = enhanced.convert("RGB")
        enhanced = enhanced.resize(original_size, Image.Resampling.LANCZOS)

        output = io.BytesIO()
        if original_format == "PNG":
            enhanced.save(output, format="PNG", optimize=True)
            extension = "png"
        elif original_format == "WEBP":
            enhanced.save(output, format="WEBP", quality=95, method=6)
            extension = "webp"
        else:
            enhanced.save(output, format="JPEG", quality=95, optimize=True)
            extension = "jpg"

        return output.getvalue(), extension, original_size

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

    batch.append({
        "file_id": photo.file_id,
        "file_name": getattr(message.document, "file_name", None) if message.document else None
    })
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

    # Clear current batch immediately so the user can start another batch later.
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
            enhanced, ext, dimensions = await asyncio.to_thread(
                enhance_image, bytes(downloaded)
            )
            await message.reply_document(
                document=io.BytesIO(enhanced),
                filename=f"enhanced_{index}.{ext}",
                caption=(
                    f"✅ Enhanced photo {index}/{len(batch)}\\n"
                    f"Original dimensions preserved: {dimensions[0]} × {dimensions[1]} px"
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
          
