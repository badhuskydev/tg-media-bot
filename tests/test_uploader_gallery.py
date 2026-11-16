"""Tests for image-post uploads: photos, albums, document fallback, caching."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.types import InputMediaDocument, InputMediaPhoto, InputMediaVideo

from src.bot.handlers import BotHandlers
from src.downloaders.ytdlp import DownloadResult
from src.services import uploader as uploader_mod
from src.services.media_cache import MediaCache
from src.services.uploader import (
    UploaderService,
    _chunk,
    _Item,
    _photo_fits,
    _plan_groups,
    _VideoProbe,
    cache_entry_from_messages,
)
from src.types.download import DownloadTask

URL = "https://x.com/jack/status/1"
CAPTION = (
    '<i><a href="https://x.com/jack/status/1">Post</a> by '
    '<a href="https://x.com/jack">Jack</a> on X</i>'
)


def _photo_msg(fid):
    return SimpleNamespace(photo=[SimpleNamespace(file_id=fid + "_small"),
                                  SimpleNamespace(file_id=fid)],
                           video=None, document=None, audio=None, animation=None)


def _video_msg(fid):
    return SimpleNamespace(photo=None, document=None, audio=None, animation=None,
                           video=SimpleNamespace(file_id=fid, width=640, height=360, duration=5))


def _doc_msg(fid):
    return SimpleNamespace(photo=None, video=None, audio=None, animation=None,
                           document=SimpleNamespace(file_id=fid))


def _album_reply(**_kw):
    """send_media_group echo: one fake message per InputMedia sent."""
    msgs = []
    for i, m in enumerate(_kw["media"]):
        fid = f"F{len(msgs)}_{i}"
        if isinstance(m, InputMediaPhoto):
            msgs.append(_photo_msg(fid))
        elif isinstance(m, InputMediaVideo):
            msgs.append(_video_msg(fid))
        else:
            msgs.append(_doc_msg(fid))
    return msgs


@pytest.fixture
def bot():
    b = MagicMock()
    b.send_photo = AsyncMock(return_value=_photo_msg("P"))
    b.send_video = AsyncMock(return_value=_video_msg("V"))
    b.send_animation = AsyncMock(return_value=MagicMock())
    b.send_document = AsyncMock(return_value=_doc_msg("D"))
    b.send_media_group = AsyncMock(side_effect=_album_reply)
    return b


@pytest.fixture(autouse=True)
def fake_probe(monkeypatch):
    """Default: every file probes as 1000x800, silent, 5s."""
    dims = {}

    async def probe(path):
        w, h = dims.get(path.name, (1000, 800))
        return _VideoProbe(width=w, height=h, has_audio=False, duration=5.0)

    monkeypatch.setattr(uploader_mod, "_probe_video", probe)
    return dims


def _files(tmp_path, *names):
    out = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"x" * 100)
        out.append(p)
    return out


async def _upload(bot, files):
    return await UploaderService(bot).upload_gallery(
        chat_id=7, files=files, source_url=URL, platform="twitter",
        author="Jack", author_url="https://x.com/jack",
    )


class TestHelpers:
    @pytest.mark.parametrize("n,sizes", [
        (1, [1]), (10, [10]), (11, [6, 5]), (20, [10, 10]), (21, [7, 7, 7]), (0, []),
    ])
    def test_chunk_balanced(self, n, sizes):
        assert [len(g) for g in _chunk(list(range(n)))] == sizes

    def test_photo_fits(self):
        assert _photo_fits(1000, 1000, 800)
        assert _photo_fits(1000, None, None)  # unknown dims: try as photo
        assert not _photo_fits(11 * 1024 * 1024, 100, 100)
        assert not _photo_fits(1000, 6000, 5000)  # w + h > 10000
        assert not _photo_fits(1000, 4200, 200)  # ratio > 20

    def test_documents_grouped_after_visual(self):
        items = [_Item("photo", "a"), _Item("document", "d"), _Item("video", "v")]
        groups = _plan_groups(items)
        assert [[i.media for i in g] for g in groups] == [["a", "v"], ["d"]]


class TestUploadGallery:
    async def test_single_image_uses_send_photo(self, bot, tmp_path):
        res = await _upload(bot, _files(tmp_path, "a.jpg"))

        assert res.complete and len(res.messages) == 1
        bot.send_media_group.assert_not_awaited()
        kwargs = bot.send_photo.await_args.kwargs
        assert kwargs["caption"] == CAPTION and kwargs["parse_mode"] == "HTML"

    async def test_album_caption_on_first_item_only(self, bot, tmp_path):
        res = await _upload(bot, _files(tmp_path, "a.jpg", "b.png", "c.webp"))

        assert res.complete and len(res.messages) == 3
        media = bot.send_media_group.await_args.kwargs["media"]
        assert all(isinstance(m, InputMediaPhoto) for m in media)
        assert media[0].caption == CAPTION and media[0].parse_mode == "HTML"
        assert media[1].caption is None and media[2].caption is None

    async def test_more_than_ten_split_into_albums(self, bot, tmp_path):
        names = [f"{i:02}.jpg" for i in range(12)]
        await _upload(bot, _files(tmp_path, *names))

        calls = bot.send_media_group.await_args_list
        assert [len(c.kwargs["media"]) for c in calls] == [6, 6]
        assert calls[0].kwargs["media"][0].caption == CAPTION
        assert calls[1].kwargs["media"][0].caption is None

    async def test_images_and_video_share_album(self, bot, tmp_path):
        await _upload(bot, _files(tmp_path, "a.jpg", "b.mp4", "c.jpg"))

        bot.send_media_group.assert_awaited_once()
        media = bot.send_media_group.await_args.kwargs["media"]
        assert [type(m) for m in media] == [InputMediaPhoto, InputMediaVideo, InputMediaPhoto]
        assert (media[1].width, media[1].height, media[1].duration) == (1000, 800, 5)

    async def test_oversized_image_sent_as_document(self, bot, tmp_path, monkeypatch):
        monkeypatch.setattr(uploader_mod, "_PHOTO_MAX_BYTES", 150)
        files = _files(tmp_path, "a.jpg", "b.jpg")
        files[1].write_bytes(b"x" * 200)  # over the (patched) photo limit

        res = await _upload(bot, files)

        assert res.complete
        # One photo alone → send_photo with the caption; the big one as a document.
        assert bot.send_photo.await_args.kwargs["caption"] == CAPTION
        assert bot.send_document.await_args.kwargs["caption"] is None

    async def test_too_large_dimensions_sent_as_document(self, bot, tmp_path, fake_probe):
        fake_probe["huge.jpg"] = (9000, 2000)
        await _upload(bot, _files(tmp_path, "a.jpg", "b.jpg", "huge.jpg"))

        media = bot.send_media_group.await_args.kwargs["media"]
        assert len(media) == 2 and all(isinstance(m, InputMediaPhoto) for m in media)
        bot.send_document.assert_awaited_once()

    async def test_rejected_album_resent_as_documents(self, bot, tmp_path):
        calls = []

        async def media_group(**kw):
            calls.append(kw["media"])
            if isinstance(kw["media"][0], InputMediaPhoto):
                raise RuntimeError("Bad Request: PHOTO_INVALID_DIMENSIONS")
            return _album_reply(**kw)

        bot.send_media_group = AsyncMock(side_effect=media_group)
        res = await _upload(bot, _files(tmp_path, "a.jpg", "b.jpg"))

        assert res.complete and len(res.messages) == 2
        assert all(isinstance(m, InputMediaDocument) for m in calls[-1])
        assert calls[-1][0].caption == CAPTION

    async def test_failure_reports_incomplete(self, bot, tmp_path):
        bot.send_media_group = AsyncMock(side_effect=RuntimeError("boom"))
        res = await _upload(bot, _files(tmp_path, "a.jpg", "b.jpg"))
        assert not res.complete and res.messages == []

    async def test_single_video_goes_through_upload_video(self, bot, tmp_path):
        # Silent 5s clip → animation, same as a normal video download.
        await _upload(bot, _files(tmp_path, "clip.mp4"))
        bot.send_animation.assert_awaited_once()
        bot.send_media_group.assert_not_awaited()

    async def test_audio_track_skipped(self, bot, tmp_path):
        # e.g. a TikTok slideshow's background music
        res = await _upload(bot, _files(tmp_path, "a.jpg", "b.jpg", "music.mp3"))
        assert res.complete
        assert len(bot.send_media_group.await_args.kwargs["media"]) == 2


class TestGalleryCache:
    def test_entry_from_album(self):
        entry = cache_entry_from_messages([_photo_msg("P1"), _video_msg("V1"), _doc_msg("D1")])
        assert entry == {"kind": "gallery", "items": [
            {"type": "photo", "file_id": "P1"},
            {"type": "video", "file_id": "V1", "width": 640, "height": 360, "duration": 5},
            {"type": "document", "file_id": "D1"},
        ]}

    def test_single_photo_entry(self):
        assert cache_entry_from_messages([_photo_msg("P1")]) == {"kind": "photo", "file_id": "P1"}

    def test_empty(self):
        assert cache_entry_from_messages([]) is None

    async def test_send_cached_regroups_albums(self, bot):
        items = [{"type": "photo", "file_id": f"P{i}"} for i in range(11)]
        items.append({"type": "document", "file_id": "D1"})
        entry = {"kind": "gallery", "items": items, "author": "Jack",
                 "author_url": "https://x.com/jack", "platform": "twitter"}

        await UploaderService(bot).send_cached(7, entry, source_url=URL, platform="unknown")

        calls = bot.send_media_group.await_args_list
        assert [len(c.kwargs["media"]) for c in calls] == [6, 5]
        assert calls[0].kwargs["media"][0].media == "P0"
        # Stored platform wins over the URL-detected one.
        assert calls[0].kwargs["media"][0].caption == CAPTION
        assert bot.send_document.await_args.kwargs["document"] == "D1"

    async def test_send_cached_photo(self, bot):
        await UploaderService(bot).send_cached(
            7, {"kind": "photo", "file_id": "P1"}, source_url=URL, platform="twitter")
        assert bot.send_photo.await_args.kwargs["photo"] == "P1"


class TestHandlerImagePost:
    """End to end through _process_download_task: gallery upload, then a
    cached resend of the same link that doesn't download again."""

    async def test_upload_then_instant_resend(self, bot, tmp_path):
        h = BotHandlers(bot=bot)
        h.settings = SimpleNamespace(temp_dir=tmp_path, upload_limit_mb=50)
        h.cache = MediaCache("")
        h.queue = MagicMock(slots_busy=MagicMock(return_value=False),
                            acquire_slot=AsyncMock())
        files = _files(tmp_path, "a.jpg", "b.jpg")
        h._download = AsyncMock(return_value=DownloadResult(
            success=True, output_path=files[0], files=files, file_size=200,
            platform="twitter", author="Jack", author_url="https://x.com/jack",
        ))

        task = DownloadTask(task_id="t1", user_id=1, url=URL)
        await h._process_download_task(task, chat_id=7, status_msg_id=None)

        media = bot.send_media_group.await_args.kwargs["media"]
        assert media[0].caption == CAPTION
        entry = h.cache.get(URL, "auto")
        assert entry["kind"] == "gallery" and len(entry["items"]) == 2
        assert entry["platform"] == "twitter" and entry["author"] == "Jack"

        bot.send_media_group.reset_mock()
        task2 = DownloadTask(task_id="t2", user_id=1, url=URL)
        await h._process_download_task(task2, chat_id=7, status_msg_id=None)

        h._download.assert_awaited_once()  # served from cache, no re-download
        resent = bot.send_media_group.await_args.kwargs["media"]
        assert [m.media for m in resent] == [i["file_id"] for i in entry["items"]]
        assert resent[0].caption == CAPTION
