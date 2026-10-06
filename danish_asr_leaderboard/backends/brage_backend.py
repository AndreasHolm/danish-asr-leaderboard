"""Backend for Harmonium/brage-v1. Eval infrastructure, not part of the model repo.

The decoding (beam 5 n-best rescored with a 4-gram and the neural LM
danish-foundation-models/munin-7b-alpha, loop guard, long-clip chunking) ships in the model
repo as brage_decode.py; this backend loads it from the snapshot, as the hviske-v6 backend
does, and passes it the harness's own normaliser, which the 4-gram scores with.

The model repo is gated: accept its terms on the model page and run `hf auth login` before
the first run. The neural LM (about 14.5 GB) is downloaded from the Hub on first use. Whisper
and the neural LM each need about 15 GiB of GPU memory and take turns on the GPU, so the
whole dataset goes to the module in one call and they swap places once per dataset; with the
model that waits on the CPU, the process uses about 22 GB of RAM. If that call fails (usually
out of memory), the dataset is redone at batch size 1; a run whose "brage settings" lines list
batch size 1 took that fallback and is not the reference decode.
"""
from __future__ import annotations

import json
import os
import sys

from danish_asr_leaderboard.audio import load_audio_array
from danish_asr_leaderboard.backends._torch_util import cuda_ok
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register

MODULE_FILE = "brage_decode.py"


class BrageBackend(Backend):
    name = "brage"
    # A failed whole-dataset call is redone in chunks of this many clips at batch size 1, so the
    # neural LM still moves onto the GPU once per chunk rather than once per clip.
    FALLBACK_CHUNK = 64

    def __init__(self, model, *, options: LoadOptions | None = None, model_ref: str = ""):
        super().__init__(model, options=options)
        self.model_ref = model_ref
        self.batch_sizes: set[int] = set()

    def _record(self, batch_size: int) -> None:
        """Print the decoding settings once per batch size (normally once per run)."""
        if batch_size in self.batch_sizes:
            return
        self.batch_sizes.add(batch_size)
        settings = {"backend": self.name, "model_ref": _home_relative(self.model_ref),
                    **self.model.describe(sorted(self.batch_sizes))}
        print("brage settings: " + json.dumps(settings, ensure_ascii=False), flush=True)

    def transcribe_batch(self, audio_paths, *, batch_size):
        self._record(batch_size)
        audios = [load_audio_array(p) for p in audio_paths]
        texts = self.model.transcribe(audios, batch_size=batch_size)
        if len(texts) != len(audio_paths):
            raise RuntimeError(f"got {len(texts)} transcripts for {len(audio_paths)} inputs")
        counts = self.model.last_counts
        if counts.get("long_clips"):
            print(f"  brage: {counts['long_clips']} long clip(s) in this batch of "
                  f"{len(audio_paths)}, cut into {counts['pieces']} pieces")
        if counts.get("greedy_fallbacks"):
            print(f"  brage: {counts['greedy_fallbacks']} clip(s) with only runaway "
                  "candidates, decoded greedily with the temperature fallback")
        return texts

    def transcribe(self, audio_paths, *, batch_size):
        """The whole dataset in one call to the module, so Whisper and the neural LM swap
        places on the GPU once. If that call fails, the dataset is redone at batch size 1 in
        chunks of FALLBACK_CHUNK clips, and a chunk that fails again one clip at a time."""
        paths = list(audio_paths)
        try:
            return self.transcribe_batch(paths, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001 - the same broad fallback as Backend.transcribe
            print(f"  WARNING: brage failed on the whole dataset ({exc}); redoing it at batch "
                  f"size 1 in chunks of {self.FALLBACK_CHUNK} clips", file=sys.stderr)
        hyps: list[str] = []
        for start in range(0, len(paths), self.FALLBACK_CHUNK):
            chunk = paths[start:start + self.FALLBACK_CHUNK]
            try:
                hyps += self.transcribe_batch(chunk, batch_size=1)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: brage failed on clips {start} to {start + len(chunk) - 1} "
                      f"({exc}); redoing them one at a time", file=sys.stderr)
                hyps += self._sequential(chunk)
        return hyps

    def transcribe_one(self, audio_path):
        return self.transcribe_batch([audio_path], batch_size=1)[0]


def _home_relative(path: str) -> str:
    """path with the user's home directory written as ~, so a settings line names no user."""
    home = os.path.expanduser("~").rstrip(os.sep)
    if path and home not in ("", "~"):
        if path == home:
            return "~"
        if path.startswith(home + os.sep):
            return "~" + path[len(home):]
    return path


def _access_denied(exc: BaseException) -> bool:
    names = {cls.__name__ for cls in type(exc).__mro__}
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return bool(names & {"GatedRepoError", "RepositoryNotFoundError"}) or status in (401, 403)


def _snapshot(model_ref: str) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model_ref)
    except Exception as exc:
        if _access_denied(exc):
            raise PermissionError(
                f"Could not download {model_ref}: the repository is gated (or does not exist). "
                f"Accept its terms at https://huggingface.co/{model_ref} while logged in, then "
                "run `hf auth login` on this machine (or set HF_TOKEN)."
            ) from exc
        raise


@register("brage")
def load(model_ref: str, options: LoadOptions) -> Backend:
    local = model_ref if os.path.isdir(model_ref) else _snapshot(model_ref)
    if not os.path.isfile(os.path.join(local, MODULE_FILE)):
        raise FileNotFoundError(
            f"{model_ref} has no {MODULE_FILE}: the brage backend runs the decoding module that "
            "ships in the Harmonium/brage-v1 model repo."
        )
    if local not in sys.path:
        sys.path.insert(0, local)
    from brage_decode import BrageASR

    from danish_asr_leaderboard.normalizer import normalise

    device = "cuda" if cuda_ok(options.device) else "cpu"
    asr = BrageASR.from_pretrained(local, device=device, normalise=normalise)
    return BrageBackend(asr, options=options, model_ref=model_ref)
