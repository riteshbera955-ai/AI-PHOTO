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
MAX_BATCH_PHOTOS = int(os.getenv("MAX_BATCH_PHOTOS", "20"))

# Reference sample background: medium neutral grey.
BACKGROUND_RGB = (126, 126, 126)
LONG_EDGE_4K = 3840

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

WELCOME = (
    "👋 BROTHERS AI 4K PHOTO ENHANCER\\n\\n"
    "Send photos one by one or as an album. When finished, send /done.\\n"
    "The bot will upscale, remove the old background, add a grey background, "
    "and return a high-resolution JPG.\\n\\n"
    "Commands: /done - process batch | /cancel - clear batch"
)

def read_output(result):
    # Replicate's current Python client returns FileOutput for image models.
    if hasattr(result, "read"):
        return result.read()
    if isinstance(result, (list, tuple)) and result:
        item = result[0]
        if hasattr(item, "read"):
            return item.read()
        if isinstance(item, (bytes, bytearray)):
            return bytes(item)
    raise RuntimeError("AI model returned an unsupported output.")

def upscale_image(image_bytes: bytes) -> bytes:
    result = replicate.run(
        "nightmareai/real-esrgan",
        input={
            "image": io.BytesIO(image_bytes),
            "scale": 4,
            "face_enhance": False
        }
    )
    return read_output(result)

def remove_background(image_bytes: bytes) -> bytes:
    result = replicate.run(
        "bria/remove-background",
        input={
            "image": io.BytesIO(image_bytes),
            "preserve_alpha": True,
            "content_moderation": False
        }
    )
    return read_output(result)

def make_4k_gray_jpg(upscaled_bytes: bytes) -> tuple[bytes, tuple[int, int]]:
    with Image.open(io.BytesIO(upscaled_bytes)) as im:
        im.load()
        rgba = ImageOps.exif_transpose(im).convert("RGBA")

        # Keep the complete subject/photo in frame; do not crop or stretch.
        w, h = rgba.size
        scale = LONG_EDGE_4K / max(w, h)
        out_w = max(1, round(w * scale))
        out_h = max(1, round(h * scale))
        rgba = rgba.resize((out_w, out_h), Image.Resampling.LANCZOS)

        background = Image.new("RGBA", (out_w, out_h), BACKGROUND_RGB + (255,))
        background.alpha_composite(rgba)
        final = background.convert("RGB")

        out = io.BytesIO()
        final.save(out, format="JPEG", quality=95, subsampling=0, optimize=True)
        return out.getvalue(), (out_w, out_h)

def process_one(raw: bytes):
    # Step 1: enlarge and enhance the full original photo.
    upscaled = upscale_image(raw)

    # Step 2: remove its original background while preserving hair/edge alpha.
    cutout = remove_background(upscaled)

    # Step 3: composite subject on neutral grey and export with long edge 3840 px.
    final_bytes, dimensions = make_4k_gray_jpg(cutout)
    return final_bytes, dimensions

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["batch"] = []
    await update.message.reply_text(WELCOME)

async def receive_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    photo = msg.photo[-1] if msg.photo else msg.document

    if not photo:
        await msg.reply_text("Please send a photo or image document.")
        return
    if photo.file_size and photo.file_size > MAX_UPLOAD_MB * 1024 * 1024:
        await msg.reply_text(f"File too large. Maximum: {MAX_UPLOAD_MB} MB.")
        return

    batch = context.user_data.setdefault("batch", [])
    if len(batch) >= MAX_BATCH_PHOTOS:
        await msg.reply_text(f"Batch limit is {MAX_BATCH_PHOTOS}. Send /done to process this batch.")
        return

    batch.append(photo.file_id)
    await msg.reply_text(
        f"📥 Added to batch: {len(batch)}/{MAX_BATCH_PHOTOS}.\\n"
        "Send more photos or type /done."
    )

async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    batch = context.user_data.get("batch", [])
    if not batch:
        await msg.reply_text("No photos in the batch. Send photos first.")
        return

    context.user_data["batch"] = []
    status = await msg.reply_text(f"⏳ Starting 4K processing for {len(batch)} photo(s)...")
    success = 0
    failed = 0

    for index, file_id in enumerate(batch, start=1):
        try:
            await status.edit_text(
                f"⏳ Processing photo {index}/{len(batch)}\\n"
                f"✅ Completed: {success} | ❌ Failed: {failed}"
            )
            tg_file = await context.bot.get_file(file_id)
            raw = await tg_file.download_as_bytearray()
            result, dimensions = await asyncio.to_thread(process_one, bytes(raw))

            await msg.reply_document(
                document=io.BytesIO(result),
                filename=f"brothers_4k_grey_{index}.jpg",
                caption=(
                    f"✅ Enhanced photo {index}/{len(batch)}\\n"
                    f"Output: {dimensions[0]} × {dimensions[1]} px\\n"
                    "Background: neutral grey"
                )
            )
            success += 1
        except Exception as exc:
            failed += 1
            log.exception("Photo %s failed: %s", index, exc)

    await status.edit_text(
        f"🏁 Batch finished!\\n✅ Successful: {success}\\n❌ Failed: {failed}\\n"
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
