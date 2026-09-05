"""The eligible universe is the alert gate, so loading it must never fail quietly.

signal_pipeline blocks any ticker outside CANDIDATE_POOL, which made a hand-written
255-ticker list the tightest constraint in the system. Screened against an independent
universe, only 3 of 2026's top 25 performers were in it -- IRON (+1268%), AEHR (+289%)
and AXTI (+268%) were never subscribed, so nothing downstream could have alerted them.

It is now file-driven. The risk that introduces is the opposite of the one it fixes: a
missing, truncated or malformed file must fall back to the built-in list rather than
narrow the gate silently.
"""
import importlib

import pytest

import src.data.dynamic_scanner as ds

BUILTIN = len(ds._DEFAULT_CANDIDATE_POOL)


def load(monkeypatch, tmp_path, text=None):
    if text is None:
        monkeypatch.setenv("UNIVERSE_FILE", str(tmp_path / "missing.txt"))
    else:
        p = tmp_path / "universe.txt"
        p.write_text(text, encoding="utf-8")
        monkeypatch.setenv("UNIVERSE_FILE", str(p))
    return importlib.reload(ds)


@pytest.fixture(autouse=True)
def restore():
    yield
    import os
    os.environ.pop("UNIVERSE_FILE", None)
    importlib.reload(ds)


class TestFallingBack:
    def test_a_missing_file_uses_the_builtin_pool(self, monkeypatch, tmp_path):
        assert len(load(monkeypatch, tmp_path).CANDIDATE_POOL) == BUILTIN

    def test_an_empty_file_uses_the_builtin_pool(self, monkeypatch, tmp_path):
        assert len(load(monkeypatch, tmp_path, "").CANDIDATE_POOL) == BUILTIN

    def test_a_comments_only_file_uses_the_builtin_pool(self, monkeypatch, tmp_path):
        m = load(monkeypatch, tmp_path, "# generated\n# nothing here\n")
        assert len(m.CANDIDATE_POOL) == BUILTIN

    def test_a_truncated_file_is_rejected(self, monkeypatch, tmp_path):
        """A half-written file would narrow the alert gate without anyone noticing --
        the exact failure this feature exists to fix, in reverse."""
        m = load(monkeypatch, tmp_path, "AAPL\nNVDA\nTSLA\n")
        assert len(m.CANDIDATE_POOL) == BUILTIN
        assert "AAPL" in m.CANDIDATE_POOL


class TestLoading:
    def test_a_larger_file_widens_the_pool(self, monkeypatch, tmp_path):
        extra = [f"ZZ{i:03d}" for i in range(BUILTIN + 50)]
        m = load(monkeypatch, tmp_path, "\n".join(extra))
        assert len(m.CANDIDATE_POOL) > BUILTIN
        assert "ZZ000" in m.CANDIDATE_POOL

    def test_the_builtin_pool_is_always_merged_in(self, monkeypatch, tmp_path):
        """Regenerating must never drop a ticker the bot already watches."""
        extra = [f"ZZ{i:03d}" for i in range(BUILTIN + 50)]
        m = load(monkeypatch, tmp_path, "\n".join(extra))
        assert set(ds._DEFAULT_CANDIDATE_POOL) <= set(m.CANDIDATE_POOL)

    def test_comments_and_blanks_are_ignored(self, monkeypatch, tmp_path):
        extra = [f"ZZ{i:03d}" for i in range(BUILTIN + 50)]
        body = "# header\n\n" + "\n".join(extra) + "\n\n# trailer\n"
        m = load(monkeypatch, tmp_path, body)
        assert "#" not in "".join(m.CANDIDATE_POOL)
        assert "" not in m.CANDIDATE_POOL

    def test_tickers_are_normalised_and_deduplicated(self, monkeypatch, tmp_path):
        extra = [f"ZZ{i:03d}" for i in range(BUILTIN + 50)]
        m = load(monkeypatch, tmp_path, "\n".join(extra + ["  aapl  ", "AAPL"]))
        assert m.CANDIDATE_POOL.count("AAPL") == 1
        assert "aapl" not in m.CANDIDATE_POOL


class TestTheGateReadsIt:
    def test_the_pipeline_gates_on_candidate_pool(self):
        import inspect

        from src.execution import signal_pipeline

        src = inspect.getsource(signal_pipeline.SignalPipeline.process_signal)
        assert "CANDIDATE_POOL" in src, \
            "the universe only matters because this gate enforces it"
