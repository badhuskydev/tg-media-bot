"""Tests for the upload flow: audio cover art, video thumbnails, animations, captions."""

import shutil
import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services import uploader as uploader_mod
from src.services.uploader import UploaderService, _VideoProbe, _parse_probe

ffmpeg_available = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not installed")

CAPTION = '<i><a href="https://youtu.be/abc">Post</a> on YouTube</i>'


def _make_jpg(path, size="450x450"):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=blue:s={size}",
         "-frames:v", "1", str(path)],
        check=True, capture_output=True,
    )


@pytest.fixture
def bot():
    b = MagicMock()
    b.send_photo = AsyncMock(return_value=MagicMock(message_id=1))
    b.send_audio = AsyncMock(return_value=MagicMock(message_id=2))
    b.send_video = AsyncMock(return_value=MagicMock(message_id=3))
    b.send_animation = AsyncMock(return_value=MagicMock(message_id=4))
    b.send_document = AsyncMock(return_value=MagicMock(message_id=5))
    return b


class TestUploadAudioCover:
    @requires_ffmpeg
    async def test_single_message_with_thumbnail_and_caption(self, bot, tmp_path):
        mp3 = tmp_path / "song.mp3"
        mp3.write_bytes(b"x" * 100)
        cover = tmp_path / "cover.jpg"
        _make_jpg(cover)

        up = UploaderService(bot)
        await up.upload_audio(
            chat_id=42, file_path=mp3, caption=CAPTION, title="Song",
            performer="Artist", duration=100, thumbnail_path=cover,
        )

        # Telegram can't combine a photo + audio in one post: a single audio
        # message carries the album-art thumbnail and the caption.
        bot.send_photo.assert_not_awaited()
        bot.send_audio.assert_awaited_once()
        audio_kwargs = bot.send_audio.await_args.kwargs
        assert audio_kwargs["caption"] == CAPTION
        assert audio_kwargs["thumbnail"] is not None
        assert audio_kwargs["performer"] == "Artist"

    async def test_no_cover_still_has_caption(self, bot, tmp_path):
        mp3 = tmp_path / "song.mp3"
        mp3.write_bytes(b"x" * 100)

        up = UploaderService(bot)
        await up.upload_audio(
            chat_id=42, file_path=mp3, caption=CAPTION, title="Song",
            thumbnail_path=None,
        )

        bot.send_photo.assert_not_awaited()
        bot.send_audio.assert_awaited_once()
        audio_kwargs = bot.send_audio.await_args.kwargs
        assert audio_kwargs["caption"] == CAPTION
        assert audio_kwargs["thumbnail"] is None


class TestUploadVideoThumbnail:
    @requires_ffmpeg
    async def test_video_sent_with_thumbnail_and_caption(self, bot, tmp_path):
        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"x" * 100)
        poster = tmp_path / "poster.jpg"
        _make_jpg(poster, "1280x720")

        up = UploaderService(bot)
        await up.upload_video(
            chat_id=7, file_path=mp4, caption=CAPTION,
            duration=120, thumbnail_path=poster,
        )

        bot.send_video.assert_awaited_once()
        kwargs = bot.send_video.await_args.kwargs
        assert kwargs["thumbnail"] is not None
        assert kwargs["duration"] == 120
        assert kwargs["caption"] == CAPTION

    async def test_video_without_thumbnail(self, bot, tmp_path):
        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"x" * 100)

        up = UploaderService(bot)
        await up.upload_video(chat_id=7, file_path=mp4, caption=CAPTION,
                              thumbnail_path=None)

        bot.send_video.assert_awaited_once()
        assert bot.send_video.await_args.kwargs["thumbnail"] is None


def _fake_probe(monkeypatch, **fields):
    async def probe(_src):
        return _VideoProbe(**fields)
    monkeypatch.setattr(uploader_mod, "_probe_video", probe)


class TestAnimation:
    async def test_short_silent_clip_sent_as_animation(self, bot, tmp_path, monkeypatch):
        mp4 = tmp_path / "loop.mp4"
        mp4.write_bytes(b"x" * 100)
        _fake_probe(monkeypatch, width=480, height=270, has_audio=False, duration=7.6)

        await UploaderService(bot).upload_video(chat_id=7, file_path=mp4, caption=CAPTION)

        bot.send_video.assert_not_awaited()
        bot.send_animation.assert_awaited_once()
        kwargs = bot.send_animation.await_args.kwargs
        assert kwargs["width"] == 480 and kwargs["height"] == 270
        assert kwargs["duration"] == 8  # probed duration fills in a missing one
        assert kwargs["caption"] == CAPTION and kwargs["parse_mode"] == "HTML"

    async def test_metadata_duration_preferred(self, bot, tmp_path, monkeypatch):
        mp4 = tmp_path / "loop.mp4"
        mp4.write_bytes(b"x" * 100)
        _fake_probe(monkeypatch, width=1, height=1, has_audio=False, duration=30)

        await UploaderService(bot).upload_video(chat_id=7, file_path=mp4, duration=12)

        assert bot.send_animation.await_args.kwargs["duration"] == 12

    async def test_clip_with_audio_stays_video(self, bot, tmp_path, monkeypatch):
        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"x" * 100)
        _fake_probe(monkeypatch, width=640, height=360, has_audio=True, duration=10)

        await UploaderService(bot).upload_video(chat_id=7, file_path=mp4)

        bot.send_animation.assert_not_awaited()
        kwargs = bot.send_video.await_args.kwargs
        assert kwargs["width"] == 640 and kwargs["height"] == 360

    async def test_long_silent_clip_stays_video(self, bot, tmp_path, monkeypatch):
        mp4 = tmp_path / "long.mp4"
        mp4.write_bytes(b"x" * 100)
        _fake_probe(monkeypatch, has_audio=False, duration=60)

        await UploaderService(bot).upload_video(chat_id=7, file_path=mp4)

        bot.send_animation.assert_not_awaited()
        bot.send_video.assert_awaited_once()

    async def test_probe_failure_stays_video(self, bot, tmp_path, monkeypatch):
        # Unknown audio/duration must not be guessed into an animation.
        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"x" * 100)
        _fake_probe(monkeypatch)

        await UploaderService(bot).upload_video(chat_id=7, file_path=mp4, duration=5)

        bot.send_animation.assert_not_awaited()
        bot.send_video.assert_awaited_once()

    async def test_source_with_audio_never_animation(self, bot, tmp_path, monkeypatch):
        # yt-dlp fell back to a video-only format: the file is silent but the
        # source has sound, so it must stay a video.
        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"x" * 100)
        _fake_probe(monkeypatch, has_audio=False, duration=10)

        await UploaderService(bot).upload_video(
            chat_id=7, file_path=mp4, source_has_audio=True,
        )

        bot.send_animation.assert_not_awaited()
        bot.send_video.assert_awaited_once()

    async def test_source_without_audio_is_animation(self, bot, tmp_path, monkeypatch):
        # Source says silent; trusted even if ffprobe couldn't tell.
        mp4 = tmp_path / "loop.mp4"
        mp4.write_bytes(b"x" * 100)
        _fake_probe(monkeypatch)

        await UploaderService(bot).upload_video(
            chat_id=7, file_path=mp4, duration=5, source_has_audio=False,
        )

        bot.send_animation.assert_awaited_once()

    async def test_file_with_audio_overrides_silent_source(self, bot, tmp_path, monkeypatch):
        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"x" * 100)
        _fake_probe(monkeypatch, has_audio=True, duration=5)

        await UploaderService(bot).upload_video(
            chat_id=7, file_path=mp4, source_has_audio=False,
        )

        bot.send_animation.assert_not_awaited()

    async def test_upload_media_passes_source_signal(self, bot, tmp_path, monkeypatch):
        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"x" * 100)
        _fake_probe(monkeypatch, has_audio=False, duration=10)

        await UploaderService(bot).upload_media(
            chat_id=7, file_path=mp4, media_format=uploader_mod.MediaFormat.AUTO,
            source_has_audio=True,
        )

        bot.send_animation.assert_not_awaited()

    async def test_animation_failure_falls_back_to_document(self, bot, tmp_path, monkeypatch):
        mp4 = tmp_path / "loop.mp4"
        mp4.write_bytes(b"x" * 100)
        _fake_probe(monkeypatch, has_audio=False, duration=5)
        bot.send_animation = AsyncMock(side_effect=RuntimeError("boom"))

        msg = await UploaderService(bot).upload_video(chat_id=7, file_path=mp4, caption=CAPTION)

        assert msg is not None
        assert bot.send_document.await_args.kwargs["caption"] == CAPTION


class TestParseProbe:
    def test_silent_video(self):
        p = _parse_probe({
            "streams": [{"codec_type": "video", "width": 640, "height": 360}],
            "format": {"duration": "4.20"},
        })
        assert (p.width, p.height, p.has_audio, p.duration) == (640, 360, False, 4.2)

    def test_with_audio_and_rotation(self):
        p = _parse_probe({
            "streams": [
                {"codec_type": "audio"},
                {"codec_type": "video", "width": 1920, "height": 1080,
                 "side_data_list": [{"rotation": -90}]},
            ],
            "format": {"duration": "N/A"},
        })
        assert (p.width, p.height) == (1080, 1920)
        assert p.has_audio is True and p.duration is None


class TestUploadMediaCaption:
    async def test_caption_built_from_author(self, bot, tmp_path, monkeypatch):
        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"x" * 100)
        _fake_probe(monkeypatch, has_audio=True, duration=100)

        await UploaderService(bot).upload_media(
            chat_id=7, file_path=mp4, media_format=uploader_mod.MediaFormat.AUTO,
            title="Ignored Title", source_url="https://youtu.be/abc", platform="youtube",
            author="Chan", author_url="https://youtube.com/@chan",
        )

        caption = bot.send_video.await_args.kwargs["caption"]
        assert caption == (
            '<i><a href="https://youtu.be/abc">Post</a> by '
            '<a href="https://youtube.com/@chan">Chan</a> on YouTube</i>'
        )


class TestSendCached:
    async def test_animation_resent_with_dimensions_and_author(self, bot):
        entry = {"kind": "animation", "file_id": "AN1", "width": 320, "height": 240,
                 "duration": 6, "author": "Chan", "author_url": "https://x.com/chan"}

        await UploaderService(bot).send_cached(
            7, entry, source_url="https://x.com/chan/status/1", platform="twitter",
        )

        kwargs = bot.send_animation.await_args.kwargs
        assert kwargs["animation"] == "AN1"
        assert (kwargs["width"], kwargs["height"], kwargs["duration"]) == (320, 240, 6)
        assert kwargs["caption"] == (
            '<i><a href="https://x.com/chan/status/1">Post</a> by '
            '<a href="https://x.com/chan">Chan</a> on X</i>'
        )

    async def test_old_entry_without_author(self, bot):
        entry = {"kind": "video", "file_id": "V1", "duration": 10}

        await UploaderService(bot).send_cached(
            7, entry, source_url="https://youtu.be/abc", platform="youtube",
        )

        assert bot.send_video.await_args.kwargs["caption"] == CAPTION
