"""Unit tests for the brage backend glue (no GPU, no network, no model download).

The decoding module that ships in the model repo is replaced by a fake brage_decode.py in a
temporary snapshot directory.
"""
import json
import sys

import numpy as np
import pytest

from danish_asr_leaderboard.backends import available_backends, load_backend
from danish_asr_leaderboard.backends import brage_backend as brage
from danish_asr_leaderboard.backends.base import LoadOptions
from danish_asr_leaderboard.normalizer import normalise

SR = 16000

FAKE_MODULE = '''
CALLS = []
FAIL_OVER = None  # a call with more clips than this raises, like running out of memory


class BrageASR:
    def __init__(self, path, device, normalise):
        self.path, self.device, self.normalise = path, device, normalise
        self.last_counts = {"clips": 0, "long_clips": 0, "pieces": 0, "greedy_fallbacks": 0}

    @classmethod
    def from_pretrained(cls, path, device="cuda", normalise=None):
        CALLS.append(("from_pretrained", path, device, normalise))
        return cls(path, device, normalise)

    def describe(self, batch_sizes=()):
        return {"model": self.path, "batch_sizes": list(batch_sizes)}

    def transcribe(self, audio, batch_size=16):
        CALLS.append(("transcribe", [len(a) // 16000 for a in audio], batch_size))
        if FAIL_OVER is not None and len(audio) > FAIL_OVER:
            raise RuntimeError("CUDA out of memory")
        long = sum(1 for a in audio if len(a) > 30 * 16000)
        self.last_counts = {"clips": len(audio), "long_clips": long, "pieces": 2 * long,
                            "greedy_fallbacks": 1 if long else 0}
        return [f"text{len(a) // 16000}" for a in audio]
'''


def test_registered():
    assert "brage" in available_backends()


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """A model directory holding a fake brage_decode.py; sys.path and sys.modules are put
    back afterwards, and CUDA counts as unavailable."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(brage, "cuda_ok", lambda device: False)
    sys.modules.pop("brage_decode", None)
    model = tmp_path / "brage-v1"
    model.mkdir()
    (model / "brage_decode.py").write_text(FAKE_MODULE, encoding="utf-8")
    yield model
    sys.modules.pop("brage_decode", None)


def _calls():
    return sys.modules["brage_decode"].CALLS


def test_a_local_model_directory_is_imported_and_given_the_harness_normaliser(snapshot):
    backend = load_backend("brage", str(snapshot), LoadOptions())
    assert isinstance(backend, brage.BrageBackend) and backend.name == "brage"
    assert _calls() == [("from_pretrained", str(snapshot), "cpu", normalise)]
    assert backend.model.normalise is normalise
    assert sys.path[0] == str(snapshot)
    load_backend("brage", str(snapshot), LoadOptions())
    assert sys.path.count(str(snapshot)) == 1  # a second load does not add it again


def test_cuda_is_used_when_available(snapshot, monkeypatch):
    monkeypatch.setattr(brage, "cuda_ok", lambda device: device == "cuda")
    load_backend("brage", str(snapshot), LoadOptions(device="cuda"))
    assert _calls()[0][2] == "cuda"


def test_a_hub_id_is_downloaded_first(snapshot, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no local directory of that name
    asked = []
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda repo_id: asked.append(repo_id) or str(snapshot))
    backend = load_backend("brage", "Harmonium/brage-v1", LoadOptions())
    assert asked == ["Harmonium/brage-v1"]
    assert _calls() == [("from_pretrained", str(snapshot), "cpu", normalise)]
    assert backend.model_ref == "Harmonium/brage-v1"


class GatedRepoError(Exception):
    """Stands in for huggingface_hub's error of the same name."""


class _HttpError(Exception):
    def __init__(self, status):
        super().__init__(f"{status} Client Error")
        self.response = type("Response", (), {"status_code": status})()


@pytest.mark.parametrize("error", [GatedRepoError("Access to model is restricted"),
                                   _HttpError(401), _HttpError(403)])
def test_a_gated_repo_says_to_accept_the_terms_and_log_in(monkeypatch, tmp_path, error):
    monkeypatch.chdir(tmp_path)

    def download(repo_id):
        raise error

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    with pytest.raises(PermissionError, match="hf auth login") as err:
        load_backend("brage", "Harmonium/brage-v1", LoadOptions())
    assert "https://huggingface.co/Harmonium/brage-v1" in str(err.value)
    assert err.value.__cause__ is error


def test_other_download_errors_pass_through(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def download(repo_id):
        raise ConnectionError("no network")

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    with pytest.raises(ConnectionError, match="no network"):
        load_backend("brage", "Harmonium/brage-v1", LoadOptions())


def test_a_model_without_the_decoding_module_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", list(sys.path))
    model = tmp_path / "whisper-without-module"
    model.mkdir()
    with pytest.raises(FileNotFoundError, match="brage_decode.py"):
        load_backend("brage", str(model), LoadOptions())
    assert str(model) not in sys.path


def _backend(snapshot, monkeypatch, seconds_by_path):
    monkeypatch.setattr(
        brage, "load_audio_array", lambda p: np.zeros(seconds_by_path[p] * SR, dtype="float32"))
    return load_backend("brage", str(snapshot), LoadOptions())


SETTINGS_LINE = "brage settings: "


def _settings_lines(out):
    return [json.loads(line[len(SETTINGS_LINE):]) for line in out.splitlines()
            if line.startswith(SETTINGS_LINE)]


def test_transcribe_batch_hands_the_harness_audio_and_batch_size_to_the_module(
        snapshot, monkeypatch, capsys):
    backend = _backend(snapshot, monkeypatch, {"a.wav": 10, "b.wav": 40, "c.wav": 5})
    assert backend.transcribe_batch(["a.wav", "b.wav", "c.wav"], batch_size=16) == [
        "text10", "text40", "text5"]
    assert _calls()[1:] == [("transcribe", [10, 40, 5], 16)]  # the batch size is passed as is
    out = capsys.readouterr().out
    assert "1 long clip(s) in this batch of 3, cut into 2 pieces" in out
    assert "1 clip(s) with only runaway candidates" in out
    assert _settings_lines(out) == [{"backend": "brage", "model_ref": str(snapshot),
                                     "model": str(snapshot), "batch_sizes": [16]}]


def test_the_settings_line_writes_the_home_directory_as_a_tilde(snapshot, monkeypatch,
                                                                capsys):
    monkeypatch.setenv("HOME", str(snapshot.parent))
    backend = _backend(snapshot, monkeypatch, {"a.wav": 10})
    backend.transcribe_batch(["a.wav"], batch_size=16)
    assert _settings_lines(capsys.readouterr().out)[0]["model_ref"] == "~/brage-v1"
    assert brage._home_relative(str(snapshot.parent) + "-other/x") == (
        str(snapshot.parent) + "-other/x")  # a longer name is not inside the home directory
    assert brage._home_relative("Harmonium/brage-v1") == "Harmonium/brage-v1"


def test_the_settings_are_printed_once_per_batch_size(snapshot, monkeypatch, capsys):
    backend = _backend(snapshot, monkeypatch, {"a.wav": 10})
    backend.transcribe_batch(["a.wav"], batch_size=16)
    backend.transcribe_one("a.wav")  # the harness's sequential fallback: batch size 1
    backend.transcribe_batch(["a.wav"], batch_size=16)
    out = capsys.readouterr().out
    assert [s["batch_sizes"] for s in _settings_lines(out)] == [[16], [1, 16]]
    assert "long clip" not in out and "runaway" not in out
    assert [c[2] for c in _calls()[1:]] == [16, 1, 16]


def test_a_short_answer_from_the_module_is_an_error(snapshot, monkeypatch):
    backend = _backend(snapshot, monkeypatch, {"a.wav": 10, "b.wav": 5})
    monkeypatch.setattr(backend.model, "transcribe", lambda audio, batch_size: ["only one"])
    with pytest.raises(RuntimeError, match="1 transcripts for 2 inputs"):
        backend.transcribe_batch(["a.wav", "b.wav"], batch_size=16)


def test_a_whole_dataset_goes_to_the_module_in_one_call(snapshot, monkeypatch, capsys):
    paths = {f"{i}.wav": 1 + i % 7 for i in range(150)}
    backend = _backend(snapshot, monkeypatch, paths)
    out = backend.transcribe(list(paths), batch_size=16)
    assert out == [f"text{s}" for s in paths.values()]
    assert _calls()[1:] == [("transcribe", list(paths.values()), 16)]  # one call, batch as is
    assert [s["batch_sizes"] for s in _settings_lines(capsys.readouterr().out)] == [[16]]


def test_a_failed_dataset_is_redone_at_batch_size_1_in_chunks(snapshot, monkeypatch, capsys):
    paths = {f"{i}.wav": 1 + i % 7 for i in range(150)}
    backend = _backend(snapshot, monkeypatch, paths)
    sys.modules["brage_decode"].FAIL_OVER = 64
    out = backend.transcribe(list(paths), batch_size=16)
    assert out == [f"text{s}" for s in paths.values()]
    calls = [(len(c[1]), c[2]) for c in _calls()[1:]]
    assert calls == [(150, 16), (64, 1), (64, 1), (22, 1)]  # one neural LM swap per chunk
    captured = capsys.readouterr()
    assert "redoing it at batch size 1 in chunks of 64 clips" in captured.err
    assert [s["batch_sizes"] for s in _settings_lines(captured.out)] == [[16], [1, 16]]


def test_a_chunk_that_fails_again_is_done_one_clip_at_a_time(snapshot, monkeypatch, capsys):
    paths = {"a.wav": 3, "b.wav": 4, "c.wav": 5}
    backend = _backend(snapshot, monkeypatch, paths)
    sys.modules["brage_decode"].FAIL_OVER = 1
    assert backend.transcribe(list(paths), batch_size=16) == ["text3", "text4", "text5"]
    calls = [(c[1], c[2]) for c in _calls()[1:]]
    assert calls == [([3, 4, 5], 16), ([3, 4, 5], 1), ([3], 1), ([4], 1), ([5], 1)]
    assert "redoing them one at a time" in capsys.readouterr().err
