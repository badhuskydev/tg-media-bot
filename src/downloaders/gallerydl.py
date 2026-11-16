"""gallery-dl downloader: fallback for image posts yt-dlp can't handle.

yt-dlp only knows video/audio, so a post with no video (a tweet with photos,
an Instagram image carousel, a Reddit gallery) fails with "No video could be
found" or similar. gallery-dl downloads the post's images (and any videos
mixed in) instead. Like the yt-dlp wrapper, it runs as a subprocess with an
argument array (no shell) into the task's temp dir.
"""

import asyncio
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import get_settings
from ..utils.logger import get_logger
from .ytdlp import DownloadResult

logger = get_logger()

# yt-dlp error substrings (lowercased) meaning "this post exists but has no
# video" — the cue to try gallery-dl. Taken from yt-dlp's extractors:
# X/Twitter, Bluesky, Tumblr ("No video could be found in this tweet/post"),
# Instagram ("There is no video in this post"), Reddit ("No media found"),
# X with a media index ("Media #2 is not a video").
_NO_VIDEO_MARKERS = (
    "no video could be found",
    "there is no video in this post",
    "no media found",
    "is not a video",
)

# TikTok photo (slideshow) posts. yt-dlp has no extractor for these and fails
# with "Unsupported URL" — also when a vm./vt.tiktok.com short link redirects
# to one (the error then names the /photo/ URL).
_TIKTOK_PHOTO_RE = re.compile(r"^https?://(?:www\.)?tiktok\.com/[^?#]*/photo/\d+", re.I)
_TIKTOK_HOST_RE = re.compile(r"^https?://(?:[\w-]+\.)*tiktok\.com/", re.I)

# A single X/Twitter post (gallery-dl's tweet extractor, minus the fx/vx mirrors).
_X_STATUS_RE = re.compile(
    r"^https?://(?:www\.|mobile\.)?(?:twitter|x)\.com/[^/?#]+/status/\d+", re.I)

# Upper bound for the metadata-only listing (has_photos); it never downloads.
_LIST_TIMEOUT = 30

_PHOTO_EXTS = {"jpg", "jpeg", "png", "webp"}

# Hard cap on files fetched for one link, so a URL that turns out to be a
# whole profile/feed can't download thousands of images.
_MAX_FILES = 50

# Per-request network stall timeout (seconds); DOWNLOAD_TIMEOUT bounds the run.
_HTTP_TIMEOUT = 60

# Sidecars/partials gallery-dl leaves next to media files.
_IGNORED_SUFFIXES = {".json", ".part", ".tmp"}


def is_no_video_error(error: str) -> bool:
    """True if a yt-dlp failure means the post has no video (try gallery-dl)."""
    e = (error or "").lower()
    return any(m in e for m in _NO_VIDEO_MARKERS)


def is_tiktok_photo_url(url: str) -> bool:
    """A full tiktok.com/@user/photo/<id> link — straight to gallery-dl."""
    return bool(_TIKTOK_PHOTO_RE.match(url or ""))


def is_tiktok_photo_error(url: str, error: str) -> bool:
    """yt-dlp failed on a TikTok link because it's a photo post — e.g. a
    vm.tiktok.com short link that redirected to /photo/ ("Unsupported URL")."""
    if not _TIKTOK_HOST_RE.match(url or ""):
        return False
    e = (error or "").lower()
    return "unsupported url" in e or "/photo/" in e


def is_x_post_url(url: str) -> bool:
    """A single X/Twitter post, which may mix videos and photos."""
    return bool(_X_STATUS_RE.match(url or ""))


def _gallery_dl_command() -> Optional[List[str]]:
    """How to invoke gallery-dl: the CLI on PATH, else this interpreter's
    module (pip install into the bot's venv). None if neither is available."""
    exe = shutil.which("gallery-dl")
    if exe:
        return [exe]
    if importlib.util.find_spec("gallery_dl") is not None:
        return [sys.executable, "-m", "gallery_dl"]
    return None


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def extract_post_meta(meta: dict) -> Tuple[str, str, str]:
    """(author name, author profile URL, platform) from gallery-dl metadata.

    gallery-dl passes through each site's own field names, so this knows the
    common ones; anything unrecognized falls back to a generic author/user
    lookup with no profile link. The platform is gallery-dl's extractor
    ``category`` (e.g. "twitter", "instagram", "reddit").
    """
    category = str(meta.get("category") or "").lower()
    author = _as_dict(meta.get("author"))
    name = handle = ""

    if category == "twitter":
        handle = author.get("name") or ""
        name = author.get("nick") or handle
        url = f"https://x.com/{handle}" if handle else ""
    elif category == "instagram":
        handle = meta.get("username") or ""
        name = meta.get("fullname") or handle
        url = f"https://www.instagram.com/{handle}/" if handle else ""
    elif category == "reddit":
        handle = meta.get("author") if isinstance(meta.get("author"), str) else ""
        if handle in ("[deleted]", "[removed]"):
            handle = ""
        name = handle
        url = f"https://www.reddit.com/user/{handle}/" if handle else ""
    elif category == "tiktok":
        handle = author.get("uniqueId") or (
            meta.get("user") if isinstance(meta.get("user"), str) else "")
        name = author.get("nickname") or handle
        url = f"https://www.tiktok.com/@{handle}" if handle else ""
    elif category == "bluesky":
        handle = author.get("handle") or ""
        name = author.get("displayName") or handle
        url = f"https://bsky.app/profile/{handle}" if handle else ""
    else:
        user = _as_dict(meta.get("user"))
        for source in (author, user):
            name = source.get("nick") or source.get("name") or source.get("username") or ""
            if name:
                break
        if not name:
            for key in ("author", "user", "username", "uploader", "artist"):
                if isinstance(meta.get(key), str) and meta[key]:
                    name = meta[key]
                    break
        url = ""

    return str(name or "").strip(), url, category


class GalleryDlDownloader:
    """Subprocess wrapper around gallery-dl for image/gallery posts."""

    def __init__(self):
        self.settings = get_settings()

    def _build_command(self, base: List[str], url: str, output_dir: Path) -> List[str]:
        cmd = list(base)
        cmd.extend([
            # Exact target dir, no per-site subdirectories.
            "--directory", str(output_dir),
            # <file>.json sidecars carrying the post/author metadata.
            "--write-metadata",
            # mtime = write time, so files sort in download (post) order.
            "--no-mtime",
            "--no-part",
            "--no-colors",
            "--range", f"1-{_MAX_FILES}",
            "--http-timeout", str(_HTTP_TIMEOUT),
            # TikTok slideshows: images only, not the background music.
            "-o", "extractor.tiktok.audio=false",
        ])
        cmd.extend(self._cookie_args())
        cmd.append(url)
        return cmd

    def _cookie_args(self) -> List[str]:
        cookies_file = self.settings.cookies_file
        if cookies_file and Path(cookies_file).exists():
            return ["--cookies", cookies_file]
        return []

    async def _run(self, cmd: List[str], timeout: float) -> Tuple[int, bytes, bytes]:
        """Run gallery-dl; (returncode, stdout, stderr). Kills the process on
        cancellation, and on timeout before raising asyncio.TimeoutError."""
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return process.returncode, stdout or b"", stderr or b""
        except BaseException:
            # Timeout, cancellation (/cancel), or a launch error: don't leave
            # gallery-dl running.
            await self._terminate(process)
            raise

    async def list_media(self, url: str) -> Optional[List[dict]]:
        """Metadata of every file in the post, without downloading anything
        (``gallery-dl --dump-json``). None if gallery-dl is missing or the
        listing fails."""
        base = _gallery_dl_command()
        if base is None:
            return None
        cmd = [*base, "--dump-json", "--no-colors",
               "--http-timeout", str(_HTTP_TIMEOUT), *self._cookie_args(), url]
        try:
            rc, stdout, stderr = await self._run(
                cmd, min(_LIST_TIMEOUT, self.settings.download_timeout))
            messages = json.loads(stdout.decode(errors="replace") or "null")
        except asyncio.TimeoutError:
            logger.info("gallery-dl listing timed out")
            return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.info(f"gallery-dl listing failed: {e}")
            return None
        if not isinstance(messages, list):
            logger.info(f"gallery-dl listing failed: {self._error_line(stderr.decode(errors='replace'))}")
            return None

        items = []
        for msg in messages:
            if not isinstance(msg, list) or not msg:
                continue
            if msg[0] == -1:  # extractor error
                logger.info(f"gallery-dl listing error: {msg[-1]}")
                return None
            if msg[0] == 3 and len(msg) >= 3 and isinstance(msg[2], dict):  # Message.Url
                items.append(msg[2])
        return items

    async def has_photos(self, url: str) -> bool:
        """True if the post at ``url`` contains at least one image. False when
        it doesn't, or when that can't be determined."""
        items = await self.list_media(url) or []
        return any(
            str(m.get("type") or "").lower() == "photo"
            or str(m.get("extension") or "").lower() in _PHOTO_EXTS
            for m in items
        )

    async def download(self, url: str, output_dir: Path) -> DownloadResult:
        """Download every file in the post at ``url`` into ``output_dir``.

        Returns a DownloadResult whose ``files`` lists the media in post order
        (``output_path`` is the first, ``file_size`` the total). A non-zero
        exit with some files still downloaded counts as success.
        """
        base = _gallery_dl_command()
        if base is None:
            return DownloadResult(success=False, error="gallery-dl is not installed")

        output_dir.mkdir(parents=True, exist_ok=True)
        existing = {p for p in output_dir.iterdir()}
        cmd = self._build_command(base, url, output_dir)

        try:
            returncode, _, stderr_b = await self._run(cmd, self.settings.download_timeout)
        except asyncio.TimeoutError:
            return DownloadResult(
                success=False,
                error=f"gallery-dl timed out after {self.settings.download_timeout}s",
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"gallery-dl exception: {e}")
            return DownloadResult(success=False, error=str(e)[:200])

        stderr = stderr_b.decode(errors="replace")
        files = self._collect_files(output_dir, existing)
        if not files:
            return DownloadResult(
                success=False,
                error=self._error_line(stderr) or "gallery-dl found no media in this post",
            )
        if returncode != 0:
            logger.warning(
                "gallery-dl exited with errors but downloaded some files",
                returncode=returncode, files=len(files),
            )

        author, author_url, platform = extract_post_meta(self._read_metadata(files[0]))
        return DownloadResult(
            success=True,
            output_path=files[0],
            files=files,
            file_size=sum(f.stat().st_size for f in files),
            platform=platform,
            author=author,
            author_url=author_url,
        )

    @staticmethod
    async def _terminate(process) -> None:
        if process and process.returncode is None:
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass

    @staticmethod
    def _error_line(stderr: str) -> str:
        """The last "[error]" line gallery-dl logged, else its last line."""
        lines = [l.strip() for l in stderr.splitlines() if l.strip()]
        errors = [l for l in lines if "[error]" in l.lower()]
        return (errors or lines or [""])[-1][:300]

    @staticmethod
    def _read_metadata(media: Path) -> dict:
        """The ``<file>.json`` sidecar --write-metadata wrote for ``media``."""
        sidecar = media.with_name(media.name + ".json")
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"No gallery-dl metadata for {media.name}: {e}")
            return {}

    @classmethod
    def _collect_files(cls, output_dir: Path, existing: set) -> List[Path]:
        """New media files in post order: gallery-dl's per-file ``num`` when
        every file has one, else download (write) time, then name."""
        files = [
            p for p in output_dir.iterdir()
            if p.is_file() and p not in existing
            and p.suffix.lower() not in _IGNORED_SUFFIXES
        ]

        def num(p: Path) -> Optional[int]:
            n = cls._read_metadata(p).get("num")
            return n if isinstance(n, int) else None

        nums = {p: num(p) for p in files}
        if files and all(n is not None for n in nums.values()):
            return sorted(files, key=lambda p: (nums[p], p.name))
        return sorted(files, key=lambda p: (p.stat().st_mtime_ns, p.name))
