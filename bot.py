import os
import io
import asyncio
import logging
from PIL import Image, ImageOps, ImageFilter
import replicate
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

BOT_TOKEN = os.getenv("BOT_TOKEN")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_BATCH_PHOTOS = int(os.getenv("MAX_BATCH_PHOTOS", "30"))

# Target canvas based on the sample. The complete photo is fitted inside it:
# no cropping and no stretching. Empty canvas areas are filled with a soft
# blurred extension of the photo.
OUT_W = 1030
OUT_H = 1536
TARGET_KB = 286

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

WELCOME = (
    "👋 Send one or more photos. When finished, send /done.\\n"
    "The bot enhances the whole image and keeps the complete frame/background. "
    "It does not intentionally crop or stretch the photo.\\n"
    "Output canvas: 1030 × 1536 JPG, around 286 KB where possible.\\n"
    "Use /cancel to clear the current batch."
)

def read_replicate_output(result):
    if hasattr(result, "read"):
        return result.read()
    if isinstance(result, (list, tuple)) and result:
        item = result[0]
        if hasattr(item, "read"):
            return item.read()
        if isinstance(item, (bytes, bytearray)):
            return bytes(item)
    raise RuntimeError("Unsupported output returned by the AI model.")

def fit_full_frame(image: Image.Image) -> Image.Image:
    """Fit the whole image into the target canvas without crop or distortion."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Fit within canvas; aspect ratio remains unchanged.
    fitted = ImageOps.contain(image, (OUT_W, OUT_H), method=Image.Resampling.LANCZOS)

    # Make a blurred, enlarged copy as a natural-looking background fill.
    bg = ImageOps.fit(image, (OUT_W, OUT_H), method=Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=28))

    x = (OUT_W - fitted.width) // 2
    y = (OUT_H - fitted.height) // 2
    bg.paste(fitted, (x, y))
    return bg

def encode_jpeg(image: Image.Image) -> bytes:
    target = TARGET_KB * 1024
    best = None
    best_gap = float("inf")
    for quality in range(95, 54, -2):
        out = io.BytesIO()
        image.save(out, "JPEG", quality=quality, optimize=True, progressive=True)
        data = out.getvalue()
        if len(data) <= target:
            return data
        gap = len(data) - target
        if gap < best_gap:
            best, best_gap = data, gap
    return best or data

def enhance_image(image_bytes: bytes):
    # Important: face_enhance is OFF to reduce invented/changed facial features.
    # Real-ESRGAN enhances the entire frame, including clothing and background.
    result = replicate.run(
        "nightmareai/real-esrgan",
        input={
            "image": io.BytesIO(image_bytes),
            "scale": 4,
            "face_enhance": False
        }
    )
    enhanced_bytes = read_replicate_output(result)
    with Image.open(io.BytesIO(enhanced_bytes)) as im:
        im.load()
        final_image = fit_full_frame(im)
        data = encode_jpeg(final_image)
    return data

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["batch"] = []
    await update.message.reply_text(WELCOME)

async def receive_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    photo = message.photo[-1] if message.photo else message.document
    if not photo:
        await message.reply_text("Please send an image.")
        return
    if photo.file_size and photo.file_size > MAX_UPLOAD_MB * 1024 * 1024:
        await message.reply_text(f"Image is too large. Maximum size: {MAX_UPLOAD_MB} MB.")
        return

    batch = context.user_data.setdefault("batch", [])
    if len(batch) >= MAX_BATCH_PHOTOS:
        await message.reply_text(f"Batch limit is {MAX_BATCH_PHOTOS}. Send /done.")
        return
    batch.append(photo.file_id)
    await message.reply_text(
        f"📥 Added: {len(batch)}/{MAX_BATCH_PHOTOS}. Send more photos or /done."
    )

async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    batch = context.user_data.get("batch", [])
    if not batch:
        await message.reply_text("No photos queued. Send photos first.")
        return
    context.user_data["batch"] = []
    status = await message.reply_text(f"⏳ Processing {len(batch)} photo(s)...")
    success = failed = 0

    for index, file_id in enumerate(batch, 1):
        try:
            await status.edit_text(
                f"⏳ Enhancing {index}/{len(batch)} | Success: {success} | Failed: {failed}"
            )
            tg_file = await context.bot.get_file(file_id)
            raw = await tg_file.download_as_bytearray()
            result = await asyncio.to_thread(enhance_image, bytes(raw))
            await message.reply_document(
                document=io.BytesIO(result),
                filename=f"brothers_enhanced_{index}.jpg",
                caption=(
                    f"✅ Photo {index}/{len(batch)} enhanced\\n"
                    f"Canvas: {OUT_W} × {OUT_H} px\\n"
                    f"File size: {len(result)/1024:.1f} KB\\n"
                    "Full frame retained; face-restoration filter disabled."
                )
            )
            success += 1
        except Exception:
            failed += 1
            log.exception("Enhancement failed for photo %s", index)

    await status.edit_text(
        f"🏁 Batch finished!\\n✅ Successful: {success}\\n❌ Failed: {failed}"
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = len(context.user_data.get("batch", []))
    context.user_data["batch"] = []
    await update.message.reply_text(f"🗑️ Cleared {n} queued photo(s).")

def main():
    if not BOT_TOKEN or not REPLICATE_API_TOKEN:
        raise RuntimeError("Set BOT_TOKEN and REPLICATE_API_TOKEN in Render Environment.")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("done", done))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, receive_photo))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
