BROTHERS AI PHOTO BOT — 1030 × 1536 Detail Update
Output settings
Exact final JPG dimensions: 1030 × 1536 pixels.
JPEG quality: 100, chroma subsampling disabled.
Portrait proportions are preserved. The full processed frame is fitted inside the target canvas; no crop or stretch is applied.
Uniform matte-grey background (#808080).
CodeFormer fidelity defaults to 0.95 to favor likeness and preserve source details.
Real-ESRGAN upscaling and restrained final sharpening are retained.
Existing features retained
/start, /done, /cancel, /status
Up to 20 queued photos by default
Sequential Replicate requests and automatic 429 retry/backoff
GPU-safe input resizing before each model
Render
Build: pip install -r requirements.txt
Start: python bot.py

Required environment variables: BOT_TOKEN, REPLICATE_API_TOKEN.
Optional: OUTPUT_WIDTH (1030), OUTPUT_HEIGHT (1536), CODEFORMER_FIDELITY (0.95), GPU_MAX_PIXELS (1800000).

Quality note
The bot preserves and enhances details available in the source. AI upscaling cannot reliably recover details that were never present in a very blurry or tiny original, and no setting can guarantee perfectly accurate reconstructed details.
