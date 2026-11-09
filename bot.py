import os
import io
import time
import base64
import asyncio
import logging
import threading

import requests
from PIL import Image, ImageOps
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
FAL_KEY = os.getenv("FAL_KEY", "").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_BATCH_PHOTOS = int(os.getenv("MAX_BATCH_PHOTOS", "30"))

# Fal.ai photo restoration endpoint. No Replicate, Gemini, NAFNet or Swin2SR.
FAL_MODEL = "fal-ai/image-editing/photo-restoration"
FAL_QUEUE_BASE = "https://queue.fal.run"
FAL_TIMEOUT_SECONDS = int(os.getenv("FAL_TIMEOUT_SECONDS", "900"))
FAL_POLL_INTERVAL = float(os.getenv("FAL_POLL_INTERVAL", "2.5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("brothers-photo-bot")
_fal_lock = threading.Lock()


def _fal_headers():
    if not FAL_KEY:
        raise RuntimeError("FAL_KEY is missing in Render Environment.")
    return {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}


def _response_json(response):
    try:
        data = response.json()
    except Exception:
        data = {"detail": response.text[:1000]}
    if not response.ok:
        message = data.get("detail") or data.get("message") or data
        raise RuntimeError(f"fal.ai HTTP {response.status_code}: {message}")
    return data


def fal_restore(image_bytes):
    """Submit a photo to fal.ai and wait for the restored image URL."""
    # fal accepts a base64 data URI, so no public image hosting is needed.
    mime = "image/jpeg"
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            mime = "image/png" if im.format == "PNG" else "image/jpeg"
    except Exception:
        pass
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_uri = f"data:{mime};base64,{encoded}"

    payload = {
        "image_url": data_uri,
        "guidance_scale": 3.5,
        "num_inference_steps": 30,
        "output_format": "jpeg"
    }

    with _fal_lock:
        submit_url = f"{FAL_QUEUE_BASE}/{FAL_MODEL}"
        submit = requests.post(
            submit_url, headers=_fal_headers(), json=payload, timeout=120
        )
        submitted = _response_json(submit)
        request_id = submitted.get("request_id")
        if not request_id:
            raise RuntimeError(f"fal.ai did not return request_id: {submitted}")

        status_url = f"{FAL_QUEUE_BASE}/{FAL_MODEL}/requests/{request_id}/status"
        result_url = f"{FAL_QUEUE_BASE}/{FAL_MODEL}/requests/{request_id}"
        deadline = time.monotonic() + FAL_TIMEOUT_SECONDS

        while time.monotonic() < deadline:
            status_response = requests.get(
                status_url, headers={"Authorization": f"Key {FAL_KEY}"}, timeout=60
            )
            status_data = _response_json(status_response)
            state = status_data.get("status", "").upper()

            if state == "COMPLETED":
                result_response = requests.get(
                    result_url, headers={"Authorization": f"Key {FAL_KEY}"}, timeout=120
                )
                result_data = _response_json(result_response)
                images = result_data.get("images") or []
                if not images or not images[0].get("url"):
                    raise RuntimeError(f"fal.ai completed but returned no image: {result_data}")
                image_response = requests.get(images[0]["url"], timeout=180)
                image_response.raise_for_status()
                return image_response.content

            if state in {"FAILED", "CANCELLED"}:
                raise RuntimeError(f"fal.ai job {state}: {status_data}")

            time.sleep(FAL_POLL_INTERVAL)

    raise TimeoutError(f"fal.ai job timed out after {FAL_TIMEOUT_SECONDS} seconds")


def normalize_output(restored_bytes, original_bytes):
    """Keep the original photo's exact pixel dimensions and avoid border bars."""
    with Image.open(io.BytesIO(original_bytes)) as original:
        original = ImageOps.exif_transpose(original)
        target_size = original.size

    with Image.open(io.BytesIO(restored_bytes)) as restored:
        restored = ImageOps.exif_transpose(restored).convert("RGB")
        # Resize directly to the original canvas. This avoids adding gray/black bars.
        if restored.size != target_size:
            restored = restored.resize(target_size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        restored.save(output, "JPEG", quality=95, subsampling=0, optimize=True)
        return output.getvalue(), target_size


def enhance_photo(original_bytes):
    restored = fal_restore(original_bytes)
    return normalize_output(restored, original_bytes)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["batch"] = []
    await update.effective_message.reply_text(
        "👋 BROTHERS AI PHOTO RESTORER\n\n"
        "Send photos, then /done to process.\n"
        "Model: fal.ai Photo Restoration.\n"
        "Original scene/background is retained; no background removal.\n"
        "Output keeps each source photo's pixel dimensions.\n\n"
        "/done — process queued photos\n"
        "/cancel — clear queue\n"
        "/status — check status"
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
    if not FAL_KEY:
        await msg.reply_text("❌ FAL_KEY is missing in Render Environment.")
        return

    processing_users = context.application.bot_data.setdefault("processing_users", set())
    user_id = update.effective_user.id
    if user_id in processing_users:
        await msg.reply_text("⏳ Your previous batch is still processing. Please wait.")
        return

    processing_users.add(user_id)
    context.user_data["batch"] = []
    status_msg = await msg.reply_text(f"⏳ Starting {len(batch)} photo(s) with fal.ai...")
    success = 0
    failed = 0
    try:
        for index, file_id in enumerate(batch, start=1):
            try:
                await status_msg.edit_text(
                    f"⏳ Processing photo {index}/{len(batch)}\n"
                    f"✅ Completed: {success} | ❌ Failed: {failed}\n"
                    "Stage: fal.ai full-photo restoration."
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
                        "Model: fal.ai Photo Restoration\n"
                        "Original background retained; no background removal."
                    )
                )
                success += 1
            except Exception as exc:
                failed += 1
                log.exception("Photo %s processing failed", index)
                safe_error = str(exc).replace(FAL_KEY, "[hidden token]")
                await msg.reply_text(f"❌ Photo {index} failed.\nError: {safe_error[:450]}")
    finally:
        processing_users.discard(user_id)

    await status_msg.edit_text(
        f"🏁 Batch finished!\n✅ Successful: {success}\n❌ Failed: {failed}\n\n"
        "Send another batch whenever ready."
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
    if not FAL_KEY:
        raise RuntimeError("FAL_KEY is missing from Render Environment.")
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
