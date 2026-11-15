"""Tests for the result-post caption builder."""

from src.services.uploader import _AUTHOR_MAX, _build_caption, _platform_label


class TestBuildCaption:
    def test_full_caption(self):
        out = _build_caption(
            "https://youtu.be/abc", "youtube",
            "Some Channel", "https://www.youtube.com/@some",
        )
        assert out == (
            '<i><a href="https://youtu.be/abc">Post</a> by '
            '<a href="https://www.youtube.com/@some">Some Channel</a> on YouTube</i>'
        )

    def test_no_author(self):
        out = _build_caption("https://x.com/u/status/1", "twitter")
        assert out == '<i><a href="https://x.com/u/status/1">Post</a> on X</i>'

    def test_author_without_url_is_unlinked(self):
        out = _build_caption("https://vimeo.com/1", "vimeo", "Jane")
        assert out == '<i><a href="https://vimeo.com/1">Post</a> by Jane on Vimeo</i>'

    def test_non_http_author_url_not_linked(self):
        # Telegram rejects the whole message for a bad href — don't emit one.
        out = _build_caption("https://vimeo.com/1", "vimeo", "Jane", "javascript:alert(1)")
        assert "javascript" not in out
        assert "by Jane on Vimeo" in out

    def test_unknown_platform_uses_domain(self):
        out = _build_caption("https://www.example.com/v/1", "unknown")
        assert out.endswith("on example.com</i>")

    def test_author_html_escaped(self):
        out = _build_caption("https://x.com/a", "twitter", "Tom & <Jerry>", "https://x.com/tj")
        assert "Tom &amp; &lt;Jerry&gt;" in out
        assert "<Jerry>" not in out

    def test_urls_escaped_in_href(self):
        out = _build_caption(
            "https://x.com/a?b=1&c=2", "twitter", "A", 'https://x.com/"q"',
        )
        assert 'href="https://x.com/a?b=1&amp;c=2"' in out
        assert 'href="https://x.com/&quot;q&quot;"' in out

    def test_long_author_truncated(self):
        out = _build_caption("https://x.com/a", "twitter", "N" * 5000)
        assert "N" * _AUTHOR_MAX not in out
        assert "N" * (_AUTHOR_MAX - 1) + "…" in out

    def test_none_without_url_or_platform(self):
        assert _build_caption(None, "") is None
        assert _build_caption("", "unknown") is None

    def test_no_url_still_names_platform(self):
        assert _build_caption(None, "youtube") == "<i>Post on YouTube</i>"


class TestPlatformLabel:
    def test_known(self):
        assert _platform_label("tiktok", "https://tiktok.com/x") == "TikTok"

    def test_unknown_domain(self):
        assert _platform_label("unknown", "https://media.example.org/x") == "media.example.org"
