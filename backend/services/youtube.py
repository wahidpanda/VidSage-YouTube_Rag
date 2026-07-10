"""YouTube helpers — transcript fetching (free) and metadata via oEmbed (free, no API key)."""
from urllib.parse import urlparse, parse_qs

import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

_LANG_PREFS = ["en", "en-US", "en-GB", "bn", "hi", "ur", "es", "fr", "de", "pt",
               "ar", "id", "ja", "ko", "ru", "tr", "vi", "zh-Hans", "zh-Hant"]


def get_video_id(url: str) -> str | None:
    """Extract the 11-char video ID from any common YouTube URL shape."""
    url = url.strip()
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").replace("www.", "").replace("m.", "")

    if host == "youtu.be":
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid or None

    if host in ("youtube.com", "youtube-nocookie.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        for prefix in ("/embed/", "/shorts/", "/live/", "/v/"):
            if parsed.path.startswith(prefix):
                part = parsed.path[len(prefix):].split("/")[0]
                return part or None
    return None


def get_transcript(video_id: str) -> tuple[list[dict], str]:
    """Return (segments, language). Each segment: {text, start, duration}."""
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=_LANG_PREFS)
    except NoTranscriptFound:
        # fall back to whatever language exists (manual first, then generated)
        listing = api.list(video_id)
        transcript = next(iter(listing))
        fetched = transcript.fetch()
    except TranscriptsDisabled:
        raise ValueError("This video has captions disabled, so it can't be analyzed.")
    except VideoUnavailable:
        raise ValueError("This video is unavailable or private.")

    segments = [
        {"text": s.text.replace("\n", " ").strip(), "start": float(s.start),
         "duration": float(s.duration)}
        for s in fetched
        if s.text and s.text.strip()
    ]
    if not segments:
        raise ValueError("No caption text was found for this video.")
    language = getattr(fetched, "language_code", "en") or "en"
    return segments, language


def get_metadata(url: str) -> dict:
    """Title / channel / thumbnail via YouTube's public oEmbed endpoint (no key needed)."""
    try:
        r = httpx.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=8,
        )
        if r.status_code == 200:
            d = r.json()
            return {
                "title": d.get("title", "Untitled video"),
                "author": d.get("author_name", ""),
                "thumbnail": d.get("thumbnail_url", ""),
            }
    except Exception:
        pass
    return {"title": "Untitled video", "author": "", "thumbnail": ""}


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
