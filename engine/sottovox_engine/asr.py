from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx_whisper
import numpy as np
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

EngineKind = Literal["sherpa-parakeet", "mlx-whisper", "faster-whisper"]

DEFAULT_PARAKEET_INT8_MODEL = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_FASTER_MODEL = "Systran/faster-whisper-base"
DEFAULT_ENGINE: EngineKind = "sherpa-parakeet"

# Empty by default: prompt text leaks into low-SNR whispered clips as hallucinations.
DEFAULT_INITIAL_PROMPT = ""


@dataclass(slots=True)
class ASRConfig:
    engine: EngineKind = DEFAULT_ENGINE
    model: str = DEFAULT_PARAKEET_INT8_MODEL
    language: str | None = "en"
    whisper_mode: bool = True
    initial_prompt: str | None = None


@dataclass(slots=True)
class ASRResult:
    text: str
    engine: str
    model: str
    language: str | None


def _hf_hub_dir() -> Path:
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return Path(env)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_local_snapshot(repo_id: str) -> str:
    """Return local HF snapshot path when complete weights exist on disk; else `repo_id`."""
    candidate = Path(repo_id)
    if candidate.exists():
        return str(candidate)

    hub_dir = _hf_hub_dir()
    model_dir = hub_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not model_dir.is_dir():
        return repo_id

    for snap in sorted(model_dir.iterdir(), reverse=True):
        if not snap.is_dir():
            continue
        has_weights = any(
            (snap / fname).exists()
            for fname in ("weights.npz", "weights.safetensors", "model.safetensors", "model.bin")
        )
        has_onnx = (snap / "encoder.int8.onnx").exists() and (snap / "tokens.txt").exists()
        if (has_weights and (snap / "config.json").exists()) or has_onnx:
            return str(snap)
    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return repo_id


def available_models() -> list[dict[str, str | bool]]:
    candidates = [
        ("sherpa-parakeet", DEFAULT_PARAKEET_INT8_MODEL),
        ("mlx-whisper", "mlx-community/whisper-large-v3-turbo"),
        ("mlx-whisper", "mlx-community/whisper-large-v3-mlx"),
        ("faster-whisper", "Systran/faster-whisper-base"),
    ]
    out: list[dict[str, str | bool]] = []
    for engine, repo in candidates:
        resolved = resolve_local_snapshot(repo)
        out.append({"engine": engine, "id": repo, "cached": resolved != repo})
    return out


def _default_model_for(engine: EngineKind) -> str:
    if engine == "sherpa-parakeet":
        return DEFAULT_PARAKEET_INT8_MODEL
    if engine == "mlx-whisper":
        return DEFAULT_MLX_MODEL
    return DEFAULT_FASTER_MODEL


class Transcriber:
    """Keeps Sherpa Parakeet / MLX / faster-whisper models warm across requests."""

    def __init__(self) -> None:
        self._faster_cache: dict[str, WhisperModel] = {}
        self._sherpa_cache: dict[str, object] = {}

    def warm(self, engine: EngineKind = DEFAULT_ENGINE, model: str | None = None) -> str:
        chosen = model or _default_model_for(engine)
        resolved = resolve_local_snapshot(chosen)
        if engine == "sherpa-parakeet":
            self._get_sherpa(resolved)
        elif engine == "mlx-whisper":
            silence = np.zeros(16_000, dtype=np.float32)
            mlx_whisper.transcribe(
                silence,
                path_or_hf_repo=resolved,
                language="en",
                condition_on_previous_text=False,
                verbose=False,
            )
        else:
            self._get_faster(resolved)
        return resolved

    def _get_sherpa(self, resolved: str):
        if resolved not in self._sherpa_cache:
            import sherpa_onnx

            snap = Path(resolved)
            if not (snap / "encoder.int8.onnx").exists():
                snap = Path(snapshot_download(resolved))
            self._sherpa_cache[resolved] = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(snap / "encoder.int8.onnx"),
                decoder=str(snap / "decoder.int8.onnx"),
                joiner=str(snap / "joiner.int8.onnx"),
                tokens=str(snap / "tokens.txt"),
                num_threads=4,
                model_type="nemo_transducer",
            )
        return self._sherpa_cache[resolved]

    def _get_faster(self, resolved: str) -> WhisperModel:
        if resolved not in self._faster_cache:
            local_only = Path(resolved).exists()
            self._faster_cache[resolved] = WhisperModel(
                resolved,
                device="cpu",
                compute_type="int8",
                local_files_only=local_only,
            )
        return self._faster_cache[resolved]

    def transcribe(self, audio_16k: np.ndarray, config: ASRConfig | None = None) -> ASRResult:
        cfg = config or ASRConfig()
        if audio_16k.size == 0:
            return ASRResult(text="", engine=cfg.engine, model=cfg.model, language=cfg.language)

        model_id = cfg.model or _default_model_for(cfg.engine)
        resolved = resolve_local_snapshot(model_id)
        lang = cfg.language if cfg.language and cfg.language != "auto" else None

        if cfg.engine == "sherpa-parakeet":
            rec = self._get_sherpa(resolved)
            stream = rec.create_stream()
            stream.accept_waveform(16_000, audio_16k.astype(np.float32, copy=False))
            rec.decode_stream(stream)
            text = (stream.result.text or "").strip()
            return ASRResult(text=text, engine=cfg.engine, model=model_id, language=lang or "en")

        # Keep whisper mode sensitive without enabling random sampling on faint breath gaps.
        no_speech_thr = 0.85 if cfg.whisper_mode else 0.60
        logprob_thr = -1.35 if cfg.whisper_mode else -1.0

        if cfg.engine == "mlx-whisper":
            result = mlx_whisper.transcribe(
                audio_16k,
                path_or_hf_repo=resolved,
                language=lang,
                initial_prompt=cfg.initial_prompt or None,
                condition_on_previous_text=False,
                no_speech_threshold=no_speech_thr,
                logprob_threshold=logprob_thr,
                compression_ratio_threshold=2.2,
                temperature=0.0,
                verbose=False,
            )
            segments = result.get("segments") or []
            if segments:
                kept = [
                    (seg.get("text") or "").strip()
                    for seg in segments
                    if not (
                        float(seg.get("avg_logprob", 0.0)) < -1.55
                        and float(seg.get("no_speech_prob", 0.0)) > 0.60
                    )
                ]
                text = " ".join(part for part in kept if part).strip()
            else:
                text = (result.get("text") or "").strip()
            detected_lang = result.get("language") or lang
            return ASRResult(text=text, engine=cfg.engine, model=model_id, language=detected_lang)

        fw_model = self._get_faster(resolved)
        segments, info = fw_model.transcribe(
            audio_16k,
            language=lang,
            initial_prompt=cfg.initial_prompt or None,
            condition_on_previous_text=False,
            no_speech_threshold=no_speech_thr,
            log_prob_threshold=logprob_thr,
            vad_filter=False,  # Our whisper DSP already handles trimming conservatively.
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        return ASRResult(
            text=text,
            engine=cfg.engine,
            model=model_id,
            language=getattr(info, "language", lang),
        )
