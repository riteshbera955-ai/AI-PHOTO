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
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

BOT_TOKEN = os.getenv('BOT_TOKEN', '').strip()
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN', '').strip()
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB', '15'))
OUTPUT_WIDTH = 1030
OUTPUT_HEIGHT = 1536
MAX_BATCH_PHOTOS = int(os.getenv('MAX_BATCH_PHOTOS', '20'))
MIN_SECONDS_BETWEEN_CALLS = float(os.getenv('MIN_SECONDS_BETWEEN_CALLS', '12'))

BRIA_MODEL = 'bria/increase-resolution'
GFPGAN_MODEL = 'tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger('brothers-photo-bot')
_call_lock = threading.Lock()
_last_call = 0.0
if REPLICATE_API_TOKEN:
    os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN


def output_to_bytes(output):
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    if hasattr(output, 'read'):
        data = output.read()
        if data:
            return data
    if isinstance(output, (list, tuple)) and output:
        return output_to_bytes(output[0])
    if isinstance(output, str) and output.startswith('http'):
        r = requests.get(output, timeout=180); r.raise_for_status(); return r.content
    url = getattr(output, 'url', None)
    if callable(url): url = url()
    if isinstance(url, str) and url.startswith('http'):
        r = requests.get(url, timeout=180); r.raise_for_status(); return r.content
    raise RuntimeError(f'Unexpected Replicate output type: {type(output).__name__}')


def run_replicate(model, inputs):
    global _last_call
    with _call_lock:
        gap = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call)
        if gap > 0:
            time.sleep(gap)
        _last_call = time.monotonic()
        return output_to_bytes(replicate.run(model, input=inputs))


def passport_crop(image, width=413, height=531):
    """Centered 35x45mm @300dpi crop. User should verify headroom before official use."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    target_ratio = width / height
    ratio = image.width / image.height
    if ratio > target_ratio:
        new_w = int(image.height * target_ratio)
        left = (image.width - new_w) // 2
        image = image.crop((left, 0, left + new_w, image.height))
    else:
        new_h = int(image.width / target_ratio)
        top = max(0, (image.height - new_h) // 2)
        image = image.crop((0, top, image.width, top + new_h))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def enhance_photo(original_bytes, make_passport=False):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        input_path = td / 'input.jpg'
        input_path.write_bytes(original_bytes)
        with Image.open(input_path) as im:
            original = ImageOps.exif_transpose(im).convert('RGB')
            normalized = td / 'normalized.jpg'
            original.save(normalized, 'JPEG', quality=96)

        # Bria preserves image content and upscales without generative re-creation.
        with open(normalized, 'rb') as f:
            bria_bytes = run_replicate(BRIA_MODEL, {
                'image': f,
                'desired_increase': 2,
                'preserve_alpha': False,
                'sync': True,
                'content_moderation': False,
            })

        bria_path = td / 'bria.png'
        bria_path.write_bytes(bria_bytes)

        # GFPGAN is applied only after full-frame upscaling; v1.4 favors identity/detail.
        with open(bria_path, 'rb') as f:
            face_bytes = run_replicate(GFPGAN_MODEL, {
                'img': f,
                'version': 'v1.4',
                'scale': 2,
            })

        with Image.open(io.BytesIO(face_bytes)) as im:
            final = ImageOps.exif_transpose(im).convert('RGB')

        if make_passport:
            final = passport_crop(final)
            dimensions = final.size
        else:
            # Retain the complete original frame and exact source dimensions; no border bars.
            # Standard output size requested by the user. For source images already
            # at 1030x1536 this does not change framing; other aspect ratios are
            # resized to the same exact canvas size.
            final = final.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)
            dimensions = (OUTPUT_WIDTH, OUTPUT_HEIGHT)

        out = io.BytesIO()
        final.save(out, 'JPEG', quality=95, subsampling=0, optimize=True)
        return out.getvalue(), dimensions


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['batch'] = []
    context.user_data['passport_mode'] = False
    await update.effective_message.reply_text(
        '👋 BROTHERS PHOTO RESTORER\n\n'
        'Send photos, then /done.\n'
        'Models: Bria Increase Resolution + GFPGAN v1.4.\n'
        'Original background retained; no background removal.\n'
        'Normal output: 1030 × 1536 px. Send images as File/Document for best quality.\n\n'
        '/passport — output 35x45mm (413x531 px) centered crop\n'
        '/normal — full frame, output 1030×1536 px\n'
        '/done — process queue\n/cancel — clear queue\n/status — status'
    )


async def passport(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['passport_mode'] = True
    await update.effective_message.reply_text('🪪 Passport mode ON: centered 35×45mm, 413×531 px. Check headroom/crop before official use.')


async def normal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['passport_mode'] = False
    await update.effective_message.reply_text('Normal mode ON. Output will be 1030 × 1536 px; full frame/background retained.')


async def receive_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    media = msg.photo[-1] if msg.photo else msg.document
    if not media: return
    if msg.document and msg.document.mime_type and not msg.document.mime_type.startswith('image/'):
        await msg.reply_text('Please send JPG, PNG or WEBP images.'); return
    if media.file_size and media.file_size > MAX_UPLOAD_MB * 1024 * 1024:
        await msg.reply_text(f'Image too large. Maximum: {MAX_UPLOAD_MB} MB.'); return
    batch = context.user_data.setdefault('batch', [])
    if len(batch) >= MAX_BATCH_PHOTOS:
        await msg.reply_text(f'Batch limit reached ({MAX_BATCH_PHOTOS}). Send /done.'); return
    batch.append(media.file_id)
    await msg.reply_text(f'📥 Added: {len(batch)}/{MAX_BATCH_PHOTOS}. Send more or /done.')


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    batch = context.user_data.get('batch', [])
    if not batch:
        await msg.reply_text('No photos queued. Send photos first.'); return
    if not REPLICATE_API_TOKEN:
        await msg.reply_text('❌ REPLICATE_API_TOKEN is missing in Render Environment.'); return
    active = context.application.bot_data.setdefault('processing_users', set())
    uid = update.effective_user.id
    if uid in active:
        await msg.reply_text('⏳ Your previous batch is still processing.'); return
    active.add(uid)
    context.user_data['batch'] = []
    passport_mode = context.user_data.get('passport_mode', False)
    status_msg = await msg.reply_text(f'⏳ Starting {len(batch)} photo(s). Models: Bria + GFPGAN...')
    success = failed = 0
    try:
        for i, file_id in enumerate(batch, 1):
            try:
                await status_msg.edit_text(f'⏳ Processing {i}/{len(batch)}\n✅ Completed: {success} | ❌ Failed: {failed}\nStage 1: Bria upscaling; Stage 2: GFPGAN face restoration.')
                tg_file = await context.bot.get_file(file_id)
                raw = bytes(await tg_file.download_as_bytearray())
                result, dims = await asyncio.to_thread(enhance_photo, raw, passport_mode)
                await msg.reply_document(
                    document=io.BytesIO(result),
                    filename=f'brothers_restored_{i}.jpg',
                    caption=(f'✅ Restored {i}/{len(batch)}\nOutput: {dims[0]} × {dims[1]} px\nModels: Bria + GFPGAN v1.4\n' + ('Passport crop: 35×45mm (centered; verify crop).' if passport_mode else 'Full frame/background retained; output 1030×1536 px; no background removal.'))
                )
                success += 1
            except Exception as exc:
                failed += 1
                log.exception('Photo %s failed', i)
                err = str(exc).replace(REPLICATE_API_TOKEN, '[hidden token]')
                await msg.reply_text(f'❌ Photo {i} failed.\nError: {err[:450]}')
    finally:
        active.discard(uid)
    await status_msg.edit_text(f'🏁 Batch finished!\n✅ Successful: {success}\n❌ Failed: {failed}')


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = len(context.user_data.get('batch', [])); context.user_data['batch'] = []
    await update.effective_message.reply_text(f'🗑️ Cleared {n} queued photo(s).')


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active = context.application.bot_data.setdefault('processing_users', set())
    await update.effective_message.reply_text(
        f"🟢 Bot running.\nQueued: {len(context.user_data.get('batch', []))}\nProcessing: {'Yes' if update.effective_user.id in active else 'No'}\nPassport mode: {'ON' if context.user_data.get('passport_mode') else 'OFF'}"
    )


async def error_handler(update, context):
    log.exception('Unhandled Telegram handler error', exc_info=context.error)


def main():
    if not BOT_TOKEN: raise RuntimeError('BOT_TOKEN is missing from Render Environment.')
    if not REPLICATE_API_TOKEN: raise RuntimeError('REPLICATE_API_TOKEN is missing from Render Environment.')
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('passport', passport))
    app.add_handler(CommandHandler('normal', normal))
    app.add_handler(CommandHandler('done', done))
    app.add_handler(CommandHandler('cancel', cancel))
    app.add_handler(CommandHandler('status', status))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, receive_photo))
    app.add_error_handler(error_handler)
    app.run_polling(drop_pending_updates=True)


if __name__ == '__main__':
    main()
