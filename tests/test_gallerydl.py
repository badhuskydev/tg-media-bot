"""Tests for the gallery-dl image-post fallback (no network)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.handlers import BotHandlers
from src.downloaders import gallerydl
from src.downloaders.gallerydl import (
    GalleryDlDownloader,
    extract_post_meta,
    is_no_video_error,
    is_tiktok_photo_error,
    is_tiktok_photo_url,
    is_x_post_url,
)
from src.downloaders.ytdlp import DownloadResult
from src.types.download import DownloadTask, MediaFormat


class TestNoVideoError:
    @pytest.mark.parametrize("err", [
        "ERROR: [twitter] 123: No video could be found in this tweet",
        "ERROR: [Instagram] abc: There is no video in this post",
        "ERROR: [Reddit] xyz: No media found",
        "ERROR: [bluesky] q: No video could be found in this post",
        "ERROR: [twitter] 123: Media #2 is not a video",
    ])
    def test_matches(self, err):
        assert is_no_video_error(err)

    @pytest.mark.parametrize("err", [
        "", None, "ERROR: Unsupported URL: https://x", "ERROR: Private video",
        "ERROR: HTTP Error 404: Not Found",
    ])
    def test_other_errors(self, err):
        assert not is_no_video_error(err)


class TestUrlHelpers:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.tiktok.com/@cat/photo/7301234567890", True),
        ("https://tiktok.com/@cat/photo/7301234567890?is_from_webapp=1", True),
        ("https://www.tiktok.com/@cat/video/7301234567890", False),
        ("https://vm.tiktok.com/ZMabc123/", False),
        ("https://example.com/tiktok.com/@cat/photo/1", False),
    ])
    def test_tiktok_photo_url(self, url, expected):
        assert is_tiktok_photo_url(url) is expected

    def test_tiktok_photo_error(self):
        short = "https://vm.tiktok.com/ZMabc123/"
        assert is_tiktok_photo_error(
            short, "ERROR: Unsupported URL: https://www.tiktok.com/@cat/photo/1")
        assert not is_tiktok_photo_error(short, "ERROR: Video not available, status code 0")
        # Not a TikTok link: "Unsupported URL" is a genuinely unsupported site.
        assert not is_tiktok_photo_error("https://example.com/x", "ERROR: Unsupported URL: x")

    @pytest.mark.parametrize("url,expected", [
        ("https://x.com/jack/status/20", True),
        ("https://twitter.com/jack/status/20?s=46", True),
        ("https://mobile.twitter.com/jack/status/20/photo/1", True),
        ("https://x.com/jack", False),
        ("https://netflix.com/jack/status/20", False),
    ])
    def test_x_post_url(self, url, expected):
        assert is_x_post_url(url) is expected


class TestExtractPostMeta:
    def test_twitter(self):
        meta = {"category": "twitter", "author": {"name": "jack", "nick": "Jack D"}}
        assert extract_post_meta(meta) == ("Jack D", "https://x.com/jack", "twitter")

    def test_instagram(self):
        meta = {"category": "instagram", "username": "nasa", "fullname": "NASA"}
        assert extract_post_meta(meta) == ("NASA", "https://www.instagram.com/nasa/", "instagram")

    def test_reddit(self):
        meta = {"category": "reddit", "author": "spez"}
        assert extract_post_meta(meta) == ("spez", "https://www.reddit.com/user/spez/", "reddit")

    def test_reddit_deleted_author(self):
        assert extract_post_meta({"category": "reddit", "author": "[deleted]"}) == ("", "", "reddit")

    def test_tiktok(self):
        meta = {"category": "tiktok", "author": {"uniqueId": "cat", "nickname": "Cat"}}
        assert extract_post_meta(meta) == ("Cat", "https://www.tiktok.com/@cat", "tiktok")

    def test_bluesky(self):
        meta = {"category": "bluesky",
                "author": {"handle": "a.bsky.social", "displayName": "A"}}
        assert extract_post_meta(meta) == ("A", "https://bsky.app/profile/a.bsky.social", "bluesky")

    def test_unknown_site_name_only(self):
        meta = {"category": "someboard", "user": {"name": "anon"}}
        assert extract_post_meta(meta) == ("anon", "", "someboard")

    def test_empty(self):
        assert extract_post_meta({}) == ("", "", "")


@pytest.fixture
def gdl(monkeypatch):
    monkeypatch.setattr(gallerydl, "_gallery_dl_command", lambda: ["gallery-dl"])
    d = GalleryDlDownloader()
    d.settings = SimpleNamespace(cookies_file="", download_timeout=30)
    return d


class TestBuildCommand:
    def test_core_flags(self, gdl, tmp_path):
        cmd = gdl._build_command(["gallery-dl"], "https://x.com/a/status/1", tmp_path)
        assert cmd[0] == "gallery-dl" and cmd[-1] == "https://x.com/a/status/1"
        assert cmd[cmd.index("--directory") + 1] == str(tmp_path)
        assert "--write-metadata" in cmd
        assert cmd[cmd.index("--range") + 1] == f"1-{gallerydl._MAX_FILES}"
        assert "--cookies" not in cmd
        # TikTok slideshow music is never downloaded.
        assert cmd[cmd.index("-o") + 1] == "extractor.tiktok.audio=false"

    def test_cookies_when_file_exists(self, gdl, tmp_path):
        cookies = tmp_path / "cookies.txt"
        cookies.write_text("# Netscape")
        gdl.settings.cookies_file = str(cookies)
        cmd = gdl._build_command(["gallery-dl"], "https://x", tmp_path)
        assert cmd[cmd.index("--cookies") + 1] == str(cookies)

    def test_missing_cookies_file_ignored(self, gdl, tmp_path):
        gdl.settings.cookies_file = str(tmp_path / "nope.txt")
        assert "--cookies" not in gdl._build_command(["gallery-dl"], "https://x", tmp_path)


class _Proc:
    def __init__(self, returncode=0, stderr=b"", stdout=b""):
        self.returncode = None
        self._rc = returncode
        self._stderr = stderr
        self._stdout = stdout

    async def communicate(self):
        self.returncode = self._rc
        return self._stdout, self._stderr

    async def wait(self):
        return self._rc

    def kill(self):
        pass


def _fake_gallery_dl(monkeypatch, files: dict, returncode=0, stderr=b""):
    """Patch the subprocess: 'downloads' {name: metadata} into --directory."""
    async def fake_exec(*cmd, **_kw):
        out = gallerydl.Path(cmd[list(cmd).index("--directory") + 1])
        for name, meta in files.items():
            (out / name).write_bytes(b"img")
            (out / (name + ".json")).write_text(json.dumps(meta))
        return _Proc(returncode, stderr)
    monkeypatch.setattr(gallerydl.asyncio, "create_subprocess_exec", fake_exec)


class TestDownload:
    async def test_files_in_post_order_with_metadata(self, gdl, tmp_path, monkeypatch):
        meta = {"category": "twitter", "author": {"name": "jack", "nick": "Jack"}}
        _fake_gallery_dl(monkeypatch, {
            "b.jpg": {**meta, "num": 2}, "a.jpg": {**meta, "num": 3},
            "c.jpg": {**meta, "num": 1},
        })
        (tmp_path / "old.mp4").write_bytes(b"leftover")  # pre-existing: ignored

        r = await gdl.download("https://x.com/jack/status/1", tmp_path)

        assert r.success
        assert [f.name for f in r.files] == ["c.jpg", "b.jpg", "a.jpg"]
        assert r.output_path.name == "c.jpg"
        assert r.file_size == 9
        assert (r.author, r.author_url, r.platform) == ("Jack", "https://x.com/jack", "twitter")

    async def test_nonzero_exit_with_files_is_success(self, gdl, tmp_path, monkeypatch):
        _fake_gallery_dl(monkeypatch, {"a.jpg": {"category": "reddit"}}, returncode=4)
        r = await gdl.download("https://reddit.com/r/x/comments/1", tmp_path)
        assert r.success and len(r.files) == 1

    async def test_no_files_reports_error_line(self, gdl, tmp_path, monkeypatch):
        _fake_gallery_dl(
            monkeypatch, {}, returncode=1,
            stderr=b"[twitter][info] hi\n[twitter][error] 404 Not Found\n",
        )
        r = await gdl.download("https://x.com/a/status/1", tmp_path)
        assert not r.success
        assert r.error == "[twitter][error] 404 Not Found"

    async def test_not_installed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gallerydl, "_gallery_dl_command", lambda: None)
        r = await GalleryDlDownloader().download("https://x", tmp_path)
        assert not r.success and "not installed" in r.error


def _fake_listing(monkeypatch, messages, returncode=0):
    """Patch the subprocess to print a --dump-json listing; returns the cmds run."""
    calls = []

    async def fake_exec(*cmd, **_kw):
        calls.append(list(cmd))
        out = messages if isinstance(messages, bytes) else json.dumps(messages).encode()
        return _Proc(returncode, stdout=out)
    monkeypatch.setattr(gallerydl.asyncio, "create_subprocess_exec", fake_exec)
    return calls


class TestListMedia:
    URL = "https://x.com/jack/status/20"

    async def test_lists_files_without_downloading(self, gdl, monkeypatch):
        calls = _fake_listing(monkeypatch, [
            [2, {"tweet_id": 20}],
            [3, "https://pbs.twimg.com/a.jpg", {"type": "photo", "extension": "jpg"}],
            [3, "https://video.twimg.com/b.mp4", {"type": "video", "extension": "mp4"}],
        ])
        items = await gdl.list_media(self.URL)
        assert [i["type"] for i in items] == ["photo", "video"]
        assert "--dump-json" in calls[0] and "--directory" not in calls[0]
        assert await gdl.has_photos(self.URL)

    async def test_videos_only(self, gdl, monkeypatch):
        _fake_listing(monkeypatch, [
            [3, "u1", {"type": "video", "extension": "mp4"}],
            [3, "u2", {"type": "animated_gif", "extension": "mp4"}],
        ])
        assert not await gdl.has_photos(self.URL)

    async def test_photo_detected_by_extension(self, gdl, monkeypatch):
        _fake_listing(monkeypatch, [[3, "u", {"extension": "png"}]])
        assert await gdl.has_photos(self.URL)

    @pytest.mark.parametrize("output", [
        [[-1, {"error": "AuthorizationError", "message": "login required"}]],
        b"not json",
        b"",
    ])
    async def test_failure_means_no_photos(self, gdl, monkeypatch, output):
        _fake_listing(monkeypatch, output, returncode=1)
        assert await gdl.list_media(self.URL) is None
        assert not await gdl.has_photos(self.URL)

    async def test_not_installed(self, gdl, monkeypatch):
        monkeypatch.setattr(gallerydl, "_gallery_dl_command", lambda: None)
        assert not await gdl.has_photos(self.URL)


class TestHandlerFallback:
    """BotHandlers._download: yt-dlp first, gallery-dl only for no-video posts."""

    @pytest.fixture
    def handlers(self):
        h = BotHandlers(bot=MagicMock())
        h.downloader.download = AsyncMock()
        h.gallery.download = AsyncMock()
        h.gallery.has_photos = AsyncMock(return_value=False)
        return h

    @staticmethod
    def _task():
        return DownloadTask(task_id="t1", user_id=1, url="https://x.com/a/status/1")

    async def test_falls_back_on_no_video(self, handlers, tmp_path):
        handlers.downloader.download.return_value = DownloadResult(
            success=False, error="ERROR: [twitter] 1: No video could be found in this tweet")
        gallery_result = DownloadResult(success=True, files=[tmp_path / "a.jpg"])
        handlers.gallery.download.return_value = gallery_result

        r = await handlers._download(self._task(), tmp_path, None)

        assert r is gallery_result
        handlers.gallery.download.assert_awaited_once_with("https://x.com/a/status/1", tmp_path)

    async def test_no_fallback_when_ytdlp_succeeds(self, handlers, tmp_path):
        ok = DownloadResult(success=True, output_path=tmp_path / "v.mp4")
        handlers.downloader.download.return_value = ok

        assert await handlers._download(self._task(), tmp_path, None) is ok
        handlers.gallery.download.assert_not_awaited()

    async def test_no_fallback_for_other_errors(self, handlers, tmp_path):
        handlers.downloader.download.return_value = DownloadResult(
            success=False, error="ERROR: Private video")

        r = await handlers._download(self._task(), tmp_path, None)

        assert r.error == "ERROR: Private video"
        handlers.gallery.download.assert_not_awaited()

    async def test_keeps_ytdlp_error_when_gallery_fails(self, handlers, tmp_path):
        original = DownloadResult(
            success=False, error="ERROR: [Instagram] x: There is no video in this post")
        handlers.downloader.download.return_value = original
        handlers.gallery.download.return_value = DownloadResult(
            success=False, error="gallery-dl is not installed")

        assert await handlers._download(self._task(), tmp_path, None) is original


class TestHandlerTikTokPhotos:
    @pytest.fixture
    def handlers(self):
        h = BotHandlers(bot=MagicMock())
        h.downloader.download = AsyncMock()
        h.gallery.download = AsyncMock()
        h.gallery.has_photos = AsyncMock(return_value=False)
        return h

    @staticmethod
    def _task(url):
        return DownloadTask(task_id="t", user_id=1, url=url)

    async def test_photo_link_skips_ytdlp(self, handlers, tmp_path):
        url = "https://www.tiktok.com/@cat/photo/7301234567890"
        ok = DownloadResult(success=True, files=[tmp_path / "1.jpg"])
        handlers.gallery.download.return_value = ok

        assert await handlers._download(self._task(url), tmp_path, None) is ok
        handlers.downloader.download.assert_not_awaited()

    async def test_short_link_to_photo_post_falls_back(self, handlers, tmp_path):
        url = "https://vm.tiktok.com/ZMabc123/"
        handlers.downloader.download.return_value = DownloadResult(
            success=False, error="ERROR: Unsupported URL: https://www.tiktok.com/@cat/photo/1")
        ok = DownloadResult(success=True, files=[tmp_path / "1.jpg"])
        handlers.gallery.download.return_value = ok

        assert await handlers._download(self._task(url), tmp_path, None) is ok
        handlers.gallery.download.assert_awaited_once_with(url, tmp_path)

    async def test_other_tiktok_errors_no_fallback(self, handlers, tmp_path):
        handlers.downloader.download.return_value = DownloadResult(
            success=False, error="ERROR: Your IP address is blocked from accessing this post")

        r = await handlers._download(
            self._task("https://www.tiktok.com/@cat/video/1"), tmp_path, None)

        assert not r.success
        handlers.gallery.download.assert_not_awaited()


class TestHandlerMixedXPost:
    URL = "https://x.com/jack/status/20"

    @pytest.fixture
    def handlers(self, tmp_path):
        h = BotHandlers(bot=MagicMock())
        self.video = DownloadResult(success=True, output_path=tmp_path / "v.mp4")
        self.post = DownloadResult(success=True, files=[tmp_path / "a.jpg", tmp_path / "b.mp4"])
        h.downloader.download = AsyncMock(return_value=self.video)
        h.gallery.download = AsyncMock(return_value=self.post)
        h.gallery.has_photos = AsyncMock(return_value=True)
        return h

    def _task(self, **kw):
        return DownloadTask(task_id="t", user_id=1, url=self.URL, **kw)

    async def test_photos_present_uses_gallery(self, handlers, tmp_path):
        assert await handlers._download(self._task(), tmp_path, None) is self.post
        handlers.gallery.has_photos.assert_awaited_once_with(self.URL)

    async def test_videos_only_keeps_ytdlp(self, handlers, tmp_path):
        handlers.gallery.has_photos.return_value = False
        assert await handlers._download(self._task(), tmp_path, None) is self.video
        handlers.gallery.download.assert_not_awaited()

    async def test_gallery_failure_keeps_videos(self, handlers, tmp_path):
        handlers.gallery.download.return_value = DownloadResult(success=False, error="429")
        assert await handlers._download(self._task(), tmp_path, None) is self.video

    @pytest.mark.parametrize("kw", [
        {"preferred_format": MediaFormat.AUDIO},
        {"max_height": 720},
    ])
    async def test_no_check_for_audio_or_quality_choice(self, handlers, tmp_path, kw):
        assert await handlers._download(self._task(**kw), tmp_path, None) is self.video
        handlers.gallery.has_photos.assert_not_called()

    async def test_no_check_for_other_sites(self, handlers, tmp_path):
        task = DownloadTask(task_id="t", user_id=1, url="https://youtu.be/abc")
        assert await handlers._download(task, tmp_path, None) is self.video
        handlers.gallery.has_photos.assert_not_called()

    async def test_check_runs_alongside_download(self, handlers, tmp_path):
        """The listing starts before yt-dlp finishes, so it adds no latency."""
        order = []

        async def has_photos(url):
            order.append("check")
            return False

        async def download(**kw):
            await asyncio.sleep(0)  # yield so the check task can start
            order.append("download")
            return self.video

        handlers.gallery.has_photos = has_photos
        handlers.downloader.download = download
        await handlers._download(self._task(), tmp_path, None)
        assert order == ["check", "download"]

    async def test_check_cancelled_when_ytdlp_fails(self, handlers, tmp_path):
        started = asyncio.Event()
        cancelled = []

        async def slow_check(url):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async def failing_download(**kw):
            await started.wait()
            return DownloadResult(success=False, error="ERROR: Private video")

        handlers.gallery.has_photos = slow_check
        handlers.downloader.download = failing_download
        r = await handlers._download(self._task(), tmp_path, None)
        await asyncio.sleep(0)
        assert not r.success and cancelled == [True]
