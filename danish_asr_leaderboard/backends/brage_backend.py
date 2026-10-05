"""Backend for Harmonium/brage-v1. Eval infrastructure, not part of the model repo.

The decoding (beam 5 n-best, 4-gram LM rescoring, loop guard, long-clip chunking) ships in
the model repo as brage_decode.py; this backend loads it from the snapshot, as the hviske-v6
backend does, and passes it the harness's own normaliser, which the LM scores with.

The model repo is gated: accept its terms on the model page and run `hf auth login` before
the first run. The submitted scores were decoded with the harness defaults
(--batch-size 16), in float16 on CUDA, which needs about 15 GiB of free GPU memory. If a batch
runs out of memory, the harness redoes the dataset one clip at a time; a run whose
"brage settings" lines list batch size 1 took that fallback and is not the reference decode.
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
