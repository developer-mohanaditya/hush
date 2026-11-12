from __future__ import annotations

import io
import math
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

TARGET_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class DSPConfig:
    """Signal-conditioning parameters tuned for close-mic whispered speech."""

    enabled: bool = True
    highpass_hz: float = 95.0
    target_rms_dbfs: float = -18.0
    max_gain_db: float = 32.0
    silence_gate_dbfs: float = -65.0
    trim_silence: bool = True
    pad_ms: int = 250


@dataclass(slots=True)
class AudioStats:
    duration_sec: float
    trimmed_duration_sec: float
    sample_rate: int
    raw_rms_dbfs: float
    raw_peak_dbfs: float
    processed_rms_dbfs: float
    applied_gain_db: float
    clipped_ratio: float
    too_quiet: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(slots=True)
class DSPResult:
    audio: np.ndarray
    sample_rate: int
    stats: AudioStats

    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, self.audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


def _to_dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-7))


def load_audio_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV/FLAC/OGG via soundfile, falling back to ffmpeg for CAF/M4A/WebM."""
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".input") as src, tempfile.NamedTemporaryFile(
            suffix=".wav"
        ) as dst:
            src.write(raw_bytes)
            src.flush()
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    src.name,
                    "-ac",
                    "1",
                    "-ar",
                    str(TARGET_SAMPLE_RATE),
                    "-f",
                    "wav",
                    dst.name,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            data, sr = sf.read(dst.name, dtype="float32", always_2d=False)

    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE or audio.size == 0:
        return audio
    ratio = Fraction(TARGET_SAMPLE_RATE, sample_rate).limit_denominator(1000)
    resampled = resample_poly(audio, ratio.numerator, ratio.denominator)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def _highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate * 0.45 or audio.size < 32:
        return audio
    sos = butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, audio)
    return np.ascontiguousarray(filtered, dtype=np.float32)


def _trim_leading_trailing_silence(
    audio: np.ndarray, sample_rate: int, pad_ms: int
) -> np.ndarray:
    """Conservatively trim long dead air while preserving soft breathy consonants."""
    if audio.size < int(sample_rate * 0.8):
        return audio
    frame = max(1, int(sample_rate * 0.02))  # 20 ms frames
    n_frames = audio.size // frame
    if n_frames < 10:
        return audio

    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    noise_floor = float(np.percentile(rms, 15))
    peak_rms = float(np.max(rms))
    if peak_rms <= max(noise_floor * 1.4, 1e-4):
        return audio

    threshold = max(noise_floor * 2.0, 1e-3)
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return audio

    pad_frames = max(1, int(pad_ms / 20))
    start = max(0, int(active[0]) - pad_frames) * frame
    end = min(audio.size, (int(active[-1]) + 1 + pad_frames) * frame)
    if end - start < int(sample_rate * 0.2):
        return audio
    return np.ascontiguousarray(audio[start:end], dtype=np.float32)


def _active_rms(audio: np.ndarray, sample_rate: int) -> float:
    if audio.size == 0:
        return 0.0
    frame = max(1, int(sample_rate * 0.02))
    n_frames = audio.size // frame
    if n_frames < 3:
        return float(np.sqrt(np.mean(audio * audio)))
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    # Use the upper 60% of frame energies so breath pauses don't pull gain down/up unevenly.
    cutoff = float(np.percentile(frame_rms, 40))
    active = frame_rms[frame_rms >= cutoff]
    return float(np.sqrt(np.mean(active * active))) if active.size else float(np.mean(frame_rms))


def _frame_modulation_ratio(audio: np.ndarray, sample_rate: int) -> float:
    """Ratio of peak 20ms frame RMS to median frame RMS (distinguishes speech from flat room hum)."""
    frame = max(1, int(sample_rate * 0.02))
    n_frames = audio.size // frame
    if n_frames < 5:
        return 10.0
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    med = float(np.median(frame_rms))
    peak = float(np.max(frame_rms))
    return peak / max(med, 1e-7)


def _is_too_quiet(audio: np.ndarray, active_db: float, sample_rate: int, gate_db: float) -> bool:
    if active_db < gate_db:
        return True
    if active_db < -56.0 and _frame_modulation_ratio(audio, sample_rate) < 2.1:
        return True
    return False


def _soft_limit(audio: np.ndarray, knee: float = 0.75) -> np.ndarray:
    """Smoothly compress peaks above `knee` into (-1, 1) without hard digital clipping."""
    out = audio.copy()
    mask = np.abs(out) > knee
    if np.any(mask):
        excess = (np.abs(out[mask]) - knee) / (1.0 - knee)
        out[mask] = np.sign(out[mask]) * (knee + (1.0 - knee) * np.tanh(excess))
    return np.clip(out, -0.99, 0.99).astype(np.float32, copy=False)


def process_audio(raw_bytes: bytes, config: DSPConfig | None = None) -> DSPResult:
    cfg = config or DSPConfig()
    audio, sr = load_audio_bytes(raw_bytes)
    audio = resample_to_16k(audio, sr)
    sr = TARGET_SAMPLE_RATE

    raw_duration = float(audio.size / sr)
    if audio.size == 0:
        stats = AudioStats(0.0, 0.0, sr, -120.0, -120.0, -120.0, 0.0, 0.0, True)
        return DSPResult(audio=audio, sample_rate=sr, stats=stats)

    raw_rms = float(np.sqrt(np.mean(audio * audio)))
    raw_peak = float(np.max(np.abs(audio)))
    clipped_ratio = float(np.mean(np.abs(audio) >= 0.985))
    raw_rms_db = _to_dbfs(raw_rms)
    raw_peak_db = _to_dbfs(raw_peak)

    # Remove DC offset regardless of DSP toggle.
    processed = (audio - float(np.mean(audio))).astype(np.float32)
    applied_gain_db = 0.0
    active_pre = _active_rms(processed, sr)
    too_quiet = _is_too_quiet(processed, _to_dbfs(active_pre), sr, cfg.silence_gate_dbfs)

    if cfg.enabled:
        processed = _highpass(processed, sr, cfg.highpass_hz)
        if cfg.trim_silence:
            processed = _trim_leading_trailing_silence(processed, sr, cfg.pad_ms)

        active = _active_rms(processed, sr)
        active_db = _to_dbfs(active)
        too_quiet = _is_too_quiet(processed, active_db, sr, cfg.silence_gate_dbfs)
        if not too_quiet and active_db < cfg.target_rms_dbfs:
            applied_gain_db = max(0.0, min(cfg.max_gain_db, cfg.target_rms_dbfs - active_db))
            gain = 10.0 ** (applied_gain_db / 20.0)
            processed = _soft_limit(processed * gain)
        else:
            processed = _soft_limit(processed)

    proc_rms_db = _to_dbfs(float(np.sqrt(np.mean(processed * processed))))
    stats = AudioStats(
        duration_sec=round(raw_duration, 3),
        trimmed_duration_sec=round(float(processed.size / sr), 3),
        sample_rate=sr,
        raw_rms_dbfs=round(raw_rms_db, 1),
        raw_peak_dbfs=round(raw_peak_db, 1),
        processed_rms_dbfs=round(proc_rms_db, 1),
        applied_gain_db=round(applied_gain_db, 1),
        clipped_ratio=round(clipped_ratio, 4),
        too_quiet=too_quiet,
    )
    return DSPResult(audio=processed, sample_rate=sr, stats=stats)
