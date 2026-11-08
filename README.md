BROTHERS AI Photo Enhancer — Bulk Edition
What it does
Accepts multiple photos in one batch.
Queue photos by sending them one by one or as a Telegram album.
Send /done when finished; the bot processes the queued photos.
Send /cancel to clear the current queue.
AI enhancement uses Replicate Real-ESRGAN with face enhancement enabled.
Preserves the exact original pixel dimensions (width × height) for every output.
Sends each enhanced image as a document to avoid extra Telegram photo compression.
Important about "same size" and 4K
This version prioritizes the exact same pixel dimensions as the input photo.
For example, a 1200 × 1600 input returns as 1200 × 1600. It enhances visual detail,
but does not enlarge that image to 4K dimensions. Exact original dimensions and
4K pixel dimensions cannot both be guaranteed when the original is smaller than 4K.

Deploy on Render
Use a Background Worker:

Build command: pip install -r requirements.txt
Start command: python bot.py
Set environment variables:

BOT_TOKEN — token from Telegram @BotFather
REPLICATE_API_TOKEN — token from https://replicate.com/account/api-tokens
MAX_UPLOAD_MB — optional, default 15
MAX_BATCH_PHOTOS — optional, default 30
Do not put API tokens in GitHub or in this README. Replicate usage is paid; check
current model pricing and your account limits. Processing is sequential to keep
the Render worker stable and avoid excessive simultaneous API requests.
