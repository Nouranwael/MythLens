from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

from faster_whisper import WhisperModel
import yt_dlp

from backend.arabic.normalization import detect_language, normalize_egyptian_arabic, summarize_key_points
from backend.claims.extraction import extract_claims


def _looks_like_url(value: str) -> bool:
    return isinstance(value, str) and bool(re.match(r"^https?://", value.strip(), re.IGNORECASE))


def _looks_like_local_media_path(value: str) -> bool:
    return isinstance(value, str) and os.path.exists(value) and os.path.isfile(value)


def _find_ffmpeg_binary() -> str:
    candidates = [os.environ.get("FFMPEG_PATH"), "ffmpeg"]
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_path = get_ffmpeg_exe()
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path
    except Exception:
        pass
    raise RuntimeError("ffmpeg was not found. Install FFmpeg or imageio-ffmpeg.")


def _download_video_from_url(url: str, output_dir: str) -> str:
    """Download one audio stream only; MythLens transcribes audio and does not need video merging."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "downloaded_media.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    matches = glob.glob(os.path.join(output_dir, "downloaded_media.*"))
    if not matches:
        raise RuntimeError(f"No media file was created for URL: {url}")
    return sorted(matches)[0]


def _extract_audio_from_video(video_path: str, output_dir: str) -> str:
    ffmpeg_path = _find_ffmpeg_binary()
    output_audio = os.path.join(output_dir, "audio.wav")
    command = [
        ffmpeg_path, "-y", "-i", str(video_path), "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(output_audio),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_audio


def _transcribe_audio(audio_path: str) -> str:
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, vad_filter=True)
    transcript = " ".join(segment.text.strip() for segment in segments if segment.text and segment.text.strip())
    return transcript.strip()


def transcribe_video(video_input: Any) -> str:
    """Transcribe actual video/audio content from a URL, local file, or raw transcript text."""
    if video_input is None:
        return ""

    if isinstance(video_input, (str, bytes)):
        if isinstance(video_input, bytes):
            text = video_input.decode("utf-8", errors="ignore")
            return text.strip()

        text = str(video_input).strip()
        if _looks_like_url(text):
            with tempfile.TemporaryDirectory(prefix="mythlens_video_") as tmp_dir:
                media_path = _download_video_from_url(text, tmp_dir)
                audio_path = _extract_audio_from_video(media_path, tmp_dir)
                return _transcribe_audio(audio_path)

        if _looks_like_local_media_path(text):
            if text.lower().endswith((".wav", ".mp3", ".m4a", ".aac")):
                return _transcribe_audio(text)
            with tempfile.TemporaryDirectory(prefix="mythlens_video_") as tmp_dir:
                audio_path = _extract_audio_from_video(text, tmp_dir)
                return _transcribe_audio(audio_path)

        return text

    if hasattr(video_input, "read"):
        content = video_input.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)

    return str(video_input)


def summarize_video_transcript(transcript: str, language: str | None = None) -> str:
    if transcript is None:
        return ""
    text = str(transcript).strip()
    if not text:
        return ""
    detected = language or detect_language(text)
    normalized = normalize_egyptian_arabic(text) if detected == "ar-EG" else text
    return summarize_key_points(normalized, language=detected, max_sentences=3)


def process_text_input(text: str) -> dict:
    if text is None:
        text = ""

    text = str(text).strip()
    if _looks_like_url(text) or _looks_like_local_media_path(text):
        transcript = transcribe_video(text)
    else:
        transcript = text

    language = detect_language(transcript)
    normalized_text = normalize_egyptian_arabic(transcript) if language == "ar-EG" else transcript
    summary = summarize_video_transcript(normalized_text, language=language)
    claims = extract_claims(normalized_text)

    return {
        "original_transcript": transcript,
        "language": language,
        "summary": summary,
        "claims": claims,
    }


def process_video_input(video_input: Any) -> dict:
    transcript = transcribe_video(video_input)
    return process_text_input(transcript)
