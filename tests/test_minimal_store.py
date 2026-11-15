"""Tests for the per-chat minimal UI mode store (on by default)."""

import json

from src.services.minimal_store import MinimalModeStore


class TestMinimalModeStore:
    def test_on_by_default(self):
        assert MinimalModeStore("").contains(123)

    def test_opt_out_and_back_in(self):
        s = MinimalModeStore("")
        s.set(123, False)
        assert not s.contains(123)
        assert s.contains(456)  # other chats unaffected
        s.set(123, True)
        assert s.contains(123)

    def test_persists_opt_outs(self, tmp_path):
        path = tmp_path / "sub" / "minimal.json"
        MinimalModeStore(str(path)).set(-100, False)
        assert json.loads(path.read_text()) == {"verbose": [-100]}
        reloaded = MinimalModeStore(str(path))
        assert not reloaded.contains(-100)
        assert reloaded.contains(1)

    def test_legacy_list_means_no_opt_outs(self, tmp_path):
        # Old format: chats that had turned minimal mode ON — now the default.
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps([1, 2]))
        s = MinimalModeStore(str(path))
        assert s.contains(1) and s.contains(2) and s.contains(3)

    def test_corrupt_file_ignored(self, tmp_path):
        path = tmp_path / "minimal.json"
        path.write_text("{bad json")
        assert MinimalModeStore(str(path)).contains(1)
