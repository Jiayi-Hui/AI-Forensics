import asyncio
import io
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from detectors import audio_detector, image_detector, text_detector, video_detector

app = FastAPI(title="AI Forensics API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str


class VideoUrlRequest(BaseModel):
    url: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Invalid image file")
    return image_detector.predict(image)


@app.post("/api/detect/text")
async def detect_text(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text is empty")
    try:
        return text_detector.predict(req.text)
    except FileNotFoundError as e:
        raise HTTPException(503, detail=str(e))


@app.post("/api/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        return audio_detector.predict(str(tmp_path))
    except FileNotFoundError as e:
        raise HTTPException(503, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        return video_detector.predict(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/detect/video-url")
async def detect_video_url(req: VideoUrlRequest):
    if not req.url.strip():
        raise HTTPException(400, "URL is empty")
    return video_detector.predict_from_url(req.url)


@app.get("/api/detect/video-url/stream")
async def stream_video_url(url: str):
    """SSE endpoint: streams step-by-step progress while processing a video URL."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def make_progress():
        def progress(step: str, msg: str, **extra):
            event = {"step": step, "msg": msg, **extra}
            loop.call_soon_threadsafe(queue.put_nowait, event)
        return progress

    def sync_work():
        p = make_progress()
        try:
            result = video_detector.predict_from_url(url, progress=p)
            loop.call_soon_threadsafe(queue.put_nowait, {"step": "complete", "msg": "分析完成", "result": result})
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"step": "error", "msg": str(e)})

    async def generate():
        future = loop.run_in_executor(None, sync_work)
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=300)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("step") in ("complete", "error"):
                    break
        finally:
            await future

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the frontend at /
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "8307demo_huijiayi"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
