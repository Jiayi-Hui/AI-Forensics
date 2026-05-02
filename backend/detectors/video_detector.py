import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from PIL import Image

from . import audio_detector, image_detector, text_detector


def _ffmpeg(*args):
    result = subprocess.run(
        ["ffmpeg", *args, "-loglevel", "error"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode().strip())


_NODE_PATHS = [
    "/usr/local/opt/node@22/bin/node",
    "/usr/local/bin/node",
    "/usr/bin/node",
    "/opt/homebrew/bin/node",
]


def _find_node() -> str | None:
    import shutil
    for p in _NODE_PATHS:
        if Path(p).exists():
            return p
    return shutil.which("node")


def download_video(url: str, out_path: str) -> None:
    import yt_dlp

    base_opts = {
        "format": "bestvideo[height<=720]+bestaudio/bestvideo[height<=720]/best[height<=720]/best",
        "outtmpl": out_path,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    # Add node runtime so yt-dlp can solve YouTube's JS signature challenges
    node_bin = _find_node()
    if node_bin:
        base_opts["js_runtimes"] = {"node": {"path": node_bin}}

    # If a cookies file was exported, use it
    _cookies_path = Path(__file__).parent / "youtube_cookies.txt"
    if _cookies_path.exists():
        base_opts["cookiefile"] = str(_cookies_path)

    with yt_dlp.YoutubeDL(base_opts) as ydl:
        ydl.download([url])


def predict_from_url(url: str, progress: Callable | None = None) -> dict:
    def emit(step, msg, **kw):
        if progress:
            progress(step, msg, **kw)

    emit("download", "正在下载视频…")
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "downloaded.mp4")
        try:
            download_video(url, video_path)
        except Exception as e:
            err_msg = str(e)
            hint = ""
            if "Sign in" in err_msg or "bot" in err_msg:
                hint = (
                    " | YouTube 需要 Cookie 认证。请将 youtube_cookies.txt 放到 "
                    "backend/detectors/ 目录（用 yt-dlp 的 --cookies-from-browser 导出）。"
                )
            emit("error", f"下载失败: {err_msg[:120]}{hint}")
            return {
                "final_label": "UNKNOWN",
                "fused_score": None,
                "error": f"Download failed: {err_msg}{hint}",
                "modalities": {},
            }
        emit("downloaded", "视频下载完成，开始多模态分析…")
        return predict(video_path, progress=progress)


def predict(video_path: str, progress: Callable | None = None) -> dict:
    def emit(step, msg, **kw):
        if progress:
            progress(step, msg, **kw)

    modalities = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── Image frames ──────────────────────────────────────────────────────
        emit("frames", "正在提取视频帧（fps=1，最多 10 帧）…")
        try:
            frame_pattern = str(tmp / "frame_%04d.jpg")
            _ffmpeg("-i", video_path, "-vf", "fps=1", "-q:v", "2", frame_pattern, "-y")
            frames = sorted(tmp.glob("frame_*.jpg"))[:10]
            if not frames:
                raise RuntimeError("No frames extracted")
            emit("frames_analyze", f"提取到 {len(frames)} 帧，正在逐帧分析（EfficientNet-B7）…")
            preds = [image_detector.predict(Image.open(str(f))) for f in frames]
            avg_fake = sum(p["fake_prob"] for p in preds) / len(preds)
            modalities["image"] = {
                "label": "FAKE" if avg_fake > 0.5 else "REAL",
                "fake_prob": round(avg_fake, 4),
                "frames_analyzed": len(preds),
                "per_frame": [{"frame": i + 1, **p} for i, p in enumerate(preds)],
            }
            emit("frames_done", f"帧分析完成 ✓  伪造概率 {round(avg_fake * 100)}%")
        except Exception as e:
            modalities["image"] = {"error": str(e)}
            emit("frames_error", f"帧分析失败: {e}")

        # ── Audio track ───────────────────────────────────────────────────────
        emit("audio", "正在分离音轨（ffmpeg → mono 16 kHz WAV）…")
        audio_path = str(tmp / "audio.wav")
        audio_ok = False
        try:
            _ffmpeg("-i", video_path, "-ac", "1", "-ar", "16000", "-vn", audio_path, "-y")
            emit("audio_analyze", "音轨分离完成，正在分析音频特征（MFCC + 随机森林）…")
            modalities["audio"] = audio_detector.predict(audio_path)
            audio_ok = True
            prob = modalities["audio"].get("prob_fake", 0)
            emit("audio_done", f"音频分析完成 ✓  合成概率 {round(prob * 100)}%")
        except Exception as e:
            modalities["audio"] = {"error": str(e)}
            emit("audio_error", f"音频分析失败: {e}")

        # ── ASR → text detection ──────────────────────────────────────────────
        emit("asr", "正在语音转文字（OpenAI Whisper base）…")
        try:
            if not audio_ok or not Path(audio_path).exists():
                raise RuntimeError("Audio extraction failed")
            import whisper
            model = whisper.load_model("base")
            transcript = model.transcribe(audio_path)["text"].strip()
            if not transcript:
                raise RuntimeError("No speech detected in audio")
            snippet = transcript[:60] + ("…" if len(transcript) > 60 else "")
            emit("text_analyze", f'ASR 完成，正在分析文本内容（CSPF-Net）："{snippet}"')
            result = text_detector.predict(transcript)
            result["transcript"] = transcript[:500]
            modalities["text"] = result
            emit("text_done", f"文本分析完成 ✓  AI 概率 {round(result['ai_probability'] * 100)}%")
        except Exception as e:
            modalities["text"] = {"error": str(e)}
            emit("text_error", f"文本分析失败: {e}")

    # ── Weighted fusion ───────────────────────────────────────────────────────
    scores, weights = [], []
    if "fake_prob" in modalities.get("image", {}):
        scores.append(modalities["image"]["fake_prob"])
        weights.append(0.5)
    if "prob_fake" in modalities.get("audio", {}):
        scores.append(modalities["audio"]["prob_fake"])
        weights.append(0.3)
    if "ai_probability" in modalities.get("text", {}):
        scores.append(modalities["text"]["ai_probability"])
        weights.append(0.2)

    if scores:
        fused = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        final_label = "FAKE / AI-GENERATED" if fused > 0.5 else "REAL / HUMAN"
    else:
        fused = None
        final_label = "UNKNOWN"

    return {
        "final_label": final_label,
        "fused_score": round(fused, 4) if fused is not None else None,
        "modalities": modalities,
    }
