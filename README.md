BROTHERS AI Photo Enhancer — Updated Bulk Edition
Features
Accepts photos sent individually or as Telegram albums/documents.
Queue up to 20 photos (configurable) and process them with /done.
/cancel clears the pending queue; /status checks that the bot is running.
Uses Replicate Real-ESRGAN for 4x upscaling with face enhancement disabled to reduce identity changes.
Uses Bria background removal, then places the subject on a solid grey #7E7E7E background.
Preserves the full frame (no intentional crop or stretch).
Exports a high-quality JPEG with the longest edge set to 3840 px by default. Portrait images remain portrait; this is 4K-class long-edge output, not a forced 3840×2160 crop.
Replicate calls are serialized with a 15-second minimum gap. HTTP 429 throttling is retried with exponential backoff and Retry-After support.
Render deployment
Use a Background Worker.

Build command: pip install -r requirements.txt
Start command: python bot.py
Set these environment variables in Render:

BOT_TOKEN — Telegram bot token from @BotFather.
REPLICATE_API_TOKEN — Replicate API token.
MAX_UPLOAD_MB — optional, default 15.
MAX_BATCH_PHOTOS — optional, default 20.
OUTPUT_LONG_EDGE — optional, default 3840.
Keep tokens in Render Environment only. Never put them in GitHub or share them in chat. Replicate model usage is paid; review current billing, usage and account rate limits. A 20-photo batch can take several minutes because the workflow calls two models per image and deliberately spaces requests to avoid throttling.

How to use
Start the bot with /start.
Send JPG, PNG or WEBP photos (as photos or image documents).
Send /done once all photos are uploaded.
Wait for the bot to return each processed image.
If the API account runs out of credit or reaches a provider-side limit, retries cannot override that account restriction; check Replicate Billing/Usage.
