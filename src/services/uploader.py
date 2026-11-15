"""Telegram upload service."""

import asyncio
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional, TypeVar
from urllib.parse import urlparse

from aiogram import Bot
from aiogram.exceptions import TelegramRetryAfter
from aiogram.types import FSInputFile as InputFile, InputMedia, Message

from ..types.download import MediaFormat
from ..utils.logger import get_logger

logger = get_logger()

# Telegram thumbnail constraints: JPEG, <200KB, max 320x320
_THUMB_MAX_BYTES = 200 * 1024

# Flood control (HTTP 429): how many times to wait out Telegram's retry_after
# and resend, and the longest single wait we'll accept before giving up.
_FLOOD_MAX_RETRIES = 3
_FLOOD_MAX_WAIT = 300

T = TypeVar("T")


async def with_flood_retry(send: Callable[[], Awaitable[T]]) -> T:
    """Run a Telegram send, waiting out flood control (429) and retrying.

    ``send`` is a zero-arg factory so each attempt builds a fresh request. A
    429 after a finished download would otherwise throw the file away; waiting
    the server-specified ``retry_after`` and resending is almost always enough.
    Re-raises once retries are exhausted or the wait is unreasonably long.
    """
    for attempt in range(_FLOOD_MAX_RETRIES + 1):
        try:
            return await send()
        except TelegramRetryAfter as e:
            if attempt >= _FLOOD_MAX_RETRIES or e.retry_after > _FLOOD_MAX_WAIT:
                raise
            logger.warning(
                f"Flood control, retrying in {e.retry_after}s",
                attempt=attempt + 1,
            )
            await asyncio.sleep(e.retry_after + 1)
    raise AssertionError("unreachable")


# Display names for platforms detected by YtDlpDownloader.detect_platform().
# Anything else ("unknown") is labelled by the source URL's domain.
_PLATFORM_NAMES = {
    "youtube": "YouTube",
    "soundcloud": "SoundCloud",
    "vimeo": "Vimeo",
    "tiktok": "TikTok",
    "twitter": "X",
    "instagram": "Instagram",
    "reddit": "Reddit",
    "twitch": "Twitch",
}

# Longest author name shown in a caption. Telegram's 1024-char caption limit
# counts visible text only (href targets don't count), so this keeps the
# caption comfortably within it.
_AUTHOR_MAX = 100

# Silent videos shorter than this are sent as animations (autoplay + loop).
_ANIMATION_MAX_SECONDS = 60


def _platform_label(platform: str, source_url: Optional[str]) -> str:
    """Human-readable platform name, falling back to the URL's domain."""
    if platform in _PLATFORM_NAMES:
        return _PLATFORM_NAMES[platform]
    host = urlparse(source_url or "").hostname or ""
    return host.removeprefix("www.")


def _is_http_url(url: Optional[str]) -> bool:
    """Telegram rejects the whole message for a non-http(s) <a href>."""
    return bool(url) and url.lower().startswith(("http://", "https://"))


def _build_caption(
    source_url: Optional[str],
    platform: str = "",
    author: str = "",
    author_url: str = "",
) -> Optional[str]:
    """Build the HTML result caption.

    ``<i><a href="SOURCE">Post</a> by <a href="AUTHOR_URL">AUTHOR</a> on PLATFORM</i>``

    Without an author it's just "Post on PLATFORM"; without an author URL the
    name is shown unlinked. Returns HTML (callers must send with
    parse_mode="HTML"), or None when there's nothing to say.
    """
    label = _platform_label(platform, source_url)

    if _is_http_url(source_url):
        post = f'<a href="{html.escape(source_url, quote=True)}">Post</a>'
    elif label:
        post = "Post"
    else:
        return None

    author = (author or "").strip()
    if len(author) > _AUTHOR_MAX:
        author = author[:_AUTHOR_MAX - 1].rstrip() + "…"
    if author:
        name = html.escape(author)
        if _is_http_url(author_url):
            name = f'<a href="{html.escape(author_url, quote=True)}">{name}</a>'
        post += f" by {name}"

    if label:
        post += f" on {html.escape(label)}"

    return f"<i>{post}</i>"


def cache_entry_from_message(message: Message) -> Optional[dict]:
    """Extract a resendable {kind, file_id, …} entry from a sent message, or
    None if it carries no cacheable file."""
    if message.audio:
        a = message.audio
        return {"kind": "audio", "file_id": a.file_id, "title": a.title or "",
                "performer": a.performer or "", "duration": a.duration or 0}
    # Checked before .document: Telegram also fills .document on animations.
    if getattr(message, "animation", None):
        a = message.animation
        return {"kind": "animation", "file_id": a.file_id,
                "width": a.width or 0, "height": a.height or 0,
                "duration": a.duration or 0}
    if message.video:
        return {"kind": "video", "file_id": message.video.file_id,
                "duration": message.video.duration or 0}
    if message.document:
        return {"kind": "document", "file_id": message.document.file_id}
    return None


@dataclass
class _VideoProbe:
    """What ffprobe tells us about a video file. ``None`` = unknown."""

    width: Optional[int] = None
    height: Optional[int] = None
    has_audio: Optional[bool] = None
    duration: Optional[float] = None


async def _probe_video(src: Path) -> _VideoProbe:
    """Probe dimensions (honoring rotation), audio presence, and duration.

    Telegram does not reliably derive dimensions from the file, so without
    them it guesses (often from the thumbnail) and renders the wrong aspect.
    Audio presence decides video vs. animation. Best-effort: on any failure
    every field is None.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error",
            "-show_entries",
            "stream=codec_type,width,height:stream_side_data=rotation"
            ":stream_tags=rotate:format=duration",
            "-of", "json", str(src),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            return _VideoProbe()
        data = json.loads(out or b"{}")
    except Exception as e:  # best-effort; upload proceeds without it
        logger.debug(f"ffprobe failed: {e}")
        return _VideoProbe()
    return _parse_probe(data)


def _parse_probe(data: dict) -> _VideoProbe:
    """Turn ffprobe's JSON output into a _VideoProbe."""
    streams = data.get("streams") or []
    probe = _VideoProbe(
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )
    try:
        dur = float((data.get("format") or {}).get("duration") or 0)
        probe.duration = dur if dur > 0 else None
    except (TypeError, ValueError):
        pass

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return probe
    w, h = video.get("width"), video.get("height")
    rot = (video.get("tags") or {}).get("rotate")
    for sd in video.get("side_data_list") or []:
        if "rotation" in sd:
            rot = sd["rotation"]
    try:
        if rot is not None and abs(int(float(rot))) % 180 == 90:
            w, h = h, w
    except (TypeError, ValueError):
        pass
    probe.width, probe.height = w, h
    return probe


def _is_animation(
    probe: _VideoProbe, duration: float, source_has_audio: Optional[bool] = None,
) -> bool:
    """A short clip whose source has no sound: send as a looping animation.

    The source's format metadata decides when it's known — a source that
    offers audio is never an animation, even if the downloaded file came out
    silent. Only when the source doesn't say (``None``) does the file's own
    audio track (ffprobe) decide. A file that does have audio is always a
    video.
    """
    dur = duration or probe.duration or 0
    if not 0 < dur < _ANIMATION_MAX_SECONDS or probe.has_audio is True:
        return False
    if source_has_audio is None:
        return probe.has_audio is False
    return source_has_audio is False


async def _prepare_thumbnail(src: Path) -> Optional[Path]:
    """Resize cover art to Telegram's thumbnail limits via ffmpeg.

    Returns a path to a <200KB JPEG no larger than 320x320, or None if the
    source is missing or ffmpeg fails. Failure is non-fatal — the audio still
    uploads without a thumbnail.
    """
    if not src or not src.exists():
        return None

    dst = src.with_name(f"{src.stem}_tg.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "scale=320:320:force_original_aspect_ratio=decrease",
        "-q:v", "5",
        str(dst),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0 or not dst.exists():
            logger.debug(f"Thumbnail resize failed: {stderr.decode(errors='replace')[:200]}")
            return None
        if dst.stat().st_size > _THUMB_MAX_BYTES:
            logger.debug("Resized thumbnail still over 200KB; skipping thumbnail")
            return None
        return dst
    except Exception as e:
        logger.debug(f"Thumbnail prep error: {e}")
        return None


class UploaderService:
    """
    Service for uploading files to Telegram.

    Handles:
    - Video uploads
    - Audio uploads
    - Captioned media
    - Progress tracking
    - Error handling
    """

    def __init__(self, bot: Bot):
        self.bot = bot

    async def upload_video(
        self,
        chat_id: int,
        file_path: Path,
        caption: Optional[str] = None,
        duration: float = 0.0,
        thumbnail_path: Optional[Path] = None,
        reply_to_message_id: Optional[int] = None,
        message_thread_id: Optional[int] = None,
        supports_streaming: bool = True,
        source_has_audio: Optional[bool] = None,
    ) -> Optional[Message]:
        """
        Upload a video to Telegram.

        Short clips (under ``_ANIMATION_MAX_SECONDS``) whose source has no
        sound are sent with send_animation instead, so they autoplay and loop
        like a GIF (see _is_animation).

        Args:
            chat_id: Target chat ID
            file_path: Path to video file
            caption: Optional HTML caption (see _build_caption)
            duration: Video length in seconds (shown by Telegram)
            thumbnail_path: Optional poster image shown before playback
            reply_to_message_id: Optional message to reply to
            message_thread_id: Optional forum topic to post into (else lands
                in the chat's "General" topic)
            supports_streaming: Enable streaming for large videos
            source_has_audio: Whether the source offers sound, from yt-dlp
                metadata (None = unknown; the file's audio track decides)

        Returns:
            Sent message or None on failure
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        thumb = await _prepare_thumbnail(thumbnail_path) if thumbnail_path else None
        probe = await _probe_video(file_path)
        as_animation = _is_animation(probe, duration, source_has_audio)
        dur = duration or probe.duration
        dur = int(round(dur)) if dur else None

        try:
            input_file = InputFile(file_path)

            if as_animation:
                message = await with_flood_retry(lambda: self.bot.send_animation(
                    chat_id=chat_id,
                    animation=input_file,
                    caption=caption,
                    parse_mode="HTML",
                    duration=dur,
                    width=probe.width,
                    height=probe.height,
                    thumbnail=InputFile(thumb) if thumb else None,
                    reply_to_message_id=reply_to_message_id,
                    message_thread_id=message_thread_id,
                ))
            else:
                message = await with_flood_retry(lambda: self.bot.send_video(
                    chat_id=chat_id,
                    video=input_file,
                    caption=caption,
                    parse_mode="HTML",
                    duration=dur,
                    width=probe.width,
                    height=probe.height,
                    thumbnail=InputFile(thumb) if thumb else None,
                    reply_to_message_id=reply_to_message_id,
                    message_thread_id=message_thread_id,
                    supports_streaming=supports_streaming,
                ))

            logger.info(
                f"{'Animation' if as_animation else 'Video'} uploaded: {file_path.name}",
                size_mb=file_path.stat().st_size / (1024 * 1024),
                message_id=message.message_id,
            )

            return message

        except TelegramRetryAfter as e:
            # Still rate-limited after retrying: a document upload would hit
            # the same per-chat limit, so don't burn another attempt on it.
            logger.error(f"Video upload failed (flood control): {e}")
            return None

        except Exception as e:
            logger.error(f"Video upload failed, trying as document: {e}")
            # Fall back to document upload for large files
            return await self.upload_document(
                chat_id=chat_id,
                file_path=file_path,
                caption=caption,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
            )

    async def upload_audio(
        self,
        chat_id: int,
        file_path: Path,
        caption: Optional[str] = None,
        title: Optional[str] = None,
        performer: Optional[str] = None,
        duration: float = 0.0,
        thumbnail_path: Optional[Path] = None,
        reply_to_message_id: Optional[int] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Message]:
        """
        Upload an audio file to Telegram.

        Args:
            chat_id: Target chat ID
            file_path: Path to audio file
            caption: Optional HTML caption (see _build_caption)
            title: Audio title (for music)
            performer: Audio performer (for music)
            duration: Track length in seconds (shown by Telegram)
            thumbnail_path: Optional cover-art image to attach
            reply_to_message_id: Optional message to reply to
            message_thread_id: Optional forum topic to post into (else lands
                in the chat's "General" topic)

        Returns:
            Sent message or None on failure
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        # One shareable message: the cover rides along as the audio's album-art
        # thumbnail (Telegram forbids combining a standalone photo with an audio
        # file in a single post/album), plus the caption.
        thumb = await _prepare_thumbnail(thumbnail_path) if thumbnail_path else None

        try:
            input_file = InputFile(file_path)

            message = await with_flood_retry(lambda: self.bot.send_audio(
                chat_id=chat_id,
                audio=input_file,
                caption=caption,
                parse_mode="HTML",
                title=title,
                performer=performer,
                duration=int(duration) if duration else None,
                thumbnail=InputFile(thumb) if thumb else None,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
            ))

            logger.info(
                f"Audio uploaded: {file_path.name}",
                size_mb=file_path.stat().st_size / (1024 * 1024),
                message_id=message.message_id,
                thumbnail=bool(thumb),
            )

            return message

        except Exception as e:
            logger.error(f"Audio upload failed: {e}")
            return None

    async def upload_document(
        self,
        chat_id: int,
        file_path: Path,
        caption: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Message]:
        """
        Upload a file as document.

        Args:
            chat_id: Target chat ID
            file_path: Path to file
            caption: Optional HTML caption (see _build_caption)
            reply_to_message_id: Optional message to reply to
            message_thread_id: Optional forum topic to post into (else lands
                in the chat's "General" topic)

        Returns:
            Sent message or None on failure
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            input_file = InputFile(file_path)

            message = await with_flood_retry(lambda: self.bot.send_document(
                chat_id=chat_id,
                document=input_file,
                caption=caption,
                parse_mode="HTML",
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
            ))

            logger.info(
                f"Document uploaded: {file_path.name}",
                size_mb=file_path.stat().st_size / (1024 * 1024),
                message_id=message.message_id,
            )

            return message

        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            return None

    async def upload_media(
        self,
        chat_id: int,
        file_path: Path,
        media_format: MediaFormat,
        title: str = "",
        performer: str = "",
        duration: float = 0.0,
        thumbnail_path: Optional[Path] = None,
        source_url: Optional[str] = None,
        platform: str = "",
        author: str = "",
        author_url: str = "",
        source_has_audio: Optional[bool] = None,
        reply_to_message_id: Optional[int] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Message]:
        """
        Upload media based on format preference.

        Automatically determines best upload method based on file extension.

        Args:
            chat_id: Target chat ID
            file_path: Path to media file
            media_format: Preferred format (video/audio/auto)
            title: Track title (audio player metadata only)
            performer: Audio performer (audio only)
            duration: Media length in seconds
            thumbnail_path: Optional cover art / poster image
            source_url: Original media URL, linked as "Post" in the caption
            platform: Platform key from detect_platform(), shown in the caption
            author: Uploader name, shown in the caption
            author_url: Uploader profile URL, linked from the author name
            source_has_audio: Whether the source offers sound (video only;
                decides video vs. animation together with the file probe)
            reply_to_message_id: Optional message to reply to
            message_thread_id: Optional forum topic to post into (else lands
                in the chat's "General" topic)

        Returns:
            Sent message or None on failure
        """
        suffix = file_path.suffix.lower()
        caption = _build_caption(source_url, platform, author, author_url)

        # Audio formats
        audio_formats = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac"}

        if suffix in audio_formats or media_format == MediaFormat.AUDIO:
            return await self.upload_audio(
                chat_id=chat_id,
                file_path=file_path,
                caption=caption,
                title=title or None,
                performer=performer or None,
                duration=duration,
                thumbnail_path=thumbnail_path,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
            )

        # Video formats
        video_formats = {".mp4", ".mkv", ".webm", ".avi", ".mov"}

        if suffix in video_formats or media_format == MediaFormat.VIDEO:
            return await self.upload_video(
                chat_id=chat_id,
                file_path=file_path,
                caption=caption,
                duration=duration,
                thumbnail_path=thumbnail_path,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
                source_has_audio=source_has_audio,
            )

        # Default to document for unknown formats
        return await self.upload_document(
            chat_id=chat_id,
            file_path=file_path,
            caption=caption,
            reply_to_message_id=reply_to_message_id,
            message_thread_id=message_thread_id,
        )

    async def send_cached(
        self,
        chat_id: int,
        entry: dict,
        source_url: Optional[str] = None,
        platform: str = "",
        reply_to_message_id: Optional[int] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Message]:
        """Resend a previously uploaded file by its cached file_id (instant).

        The caption is rebuilt from the entry's stored author, so a resent
        post looks the same as the original.

        Raises on failure (e.g. a stale file_id) so the caller can fall back to
        a fresh download.
        """
        kind = entry["kind"]
        file_id = entry["file_id"]
        caption = _build_caption(
            source_url, platform,
            entry.get("author", ""), entry.get("author_url", ""),
        )
        duration = int(entry["duration"]) if entry.get("duration") else None

        if kind == "audio":
            return await with_flood_retry(lambda: self.bot.send_audio(
                chat_id=chat_id, audio=file_id, caption=caption, parse_mode="HTML",
                title=entry.get("title") or None, performer=entry.get("performer") or None,
                duration=duration, reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
            ))
        if kind == "animation":
            return await with_flood_retry(lambda: self.bot.send_animation(
                chat_id=chat_id, animation=file_id, caption=caption, parse_mode="HTML",
                duration=duration, width=entry.get("width") or None,
                height=entry.get("height") or None,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
            ))
        if kind == "video":
            return await with_flood_retry(lambda: self.bot.send_video(
                chat_id=chat_id, video=file_id, caption=caption, parse_mode="HTML",
                duration=duration, supports_streaming=True,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
            ))
        return await with_flood_retry(lambda: self.bot.send_document(
            chat_id=chat_id, document=file_id, caption=caption, parse_mode="HTML",
            reply_to_message_id=reply_to_message_id,
            message_thread_id=message_thread_id,
        ))

    async def send_progress_message(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
        message_thread_id: Optional[int] = None,
    ) -> Optional[Message]:
        """Send a simple progress/status message."""
        try:
            return await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message_thread_id,
            )
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None

    async def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
    ) -> bool:
        """Edit an existing message."""
        try:
            await self.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to edit message: {e}")
            return False
