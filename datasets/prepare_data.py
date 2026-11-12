"""Download open English speech & hallucination probes and synthesize Hush whisper benchmark conditions."""

from __future__ import annotations

import io
import json
import pathlib
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import butter, lfilter, sosfilt

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
TARGET_SR = 16000


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def _scale_to_dbfs(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    cur = _rms(audio)
    if cur < 1e-8:
        return audio.astype(np.float32)
    target_rms = 10.0 ** (target_dbfs / 20.0)
    scaled = audio * (target_rms / cur)
    return np.clip(scaled, -0.98, 0.98).astype(np.float32)


def _pinkish_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n_samples).astype(np.float32)
    b, a = butter(1, 0.08, btype="low")
    shaped = lfilter(b, a, white).astype(np.float32)
    return shaped / (_rms(shaped) + 1e-8)


def _synthesize_pseudo_whisper(clean: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Convert modal speech into unvoiced whisper-like breath speech via frame-wise LPC excitation."""
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    order = 14
    out = np.zeros_like(clean, dtype=np.float32)
    win = np.hanning(frame_len).astype(np.float32)

    # Slight high-pass removes fundamental pitch (F0)
    hp_sos = butter(3, 240.0, btype="highpass", fs=sr, output="sos")
    padded = np.pad(clean, (0, frame_len))

    for start in range(0, len(clean), hop_len):
        frame = padded[start : start + frame_len] * win
        energy = _rms(frame)
        if energy < 1e-4:
            continue
        try:
            a_coeffs = librosa.lpc(frame, order=order)
            # Filter Gaussian noise through vocal-tract all-pole envelope to strip F0 harmonics
            noise = rng.standard_normal(frame_len).astype(np.float32)
            unvoiced = lfilter([1.0], a_coeffs, noise).astype(np.float32)
            unvoiced = unvoiced / (_rms(unvoiced) + 1e-8) * energy * win
        except Exception:
            unvoiced = frame
        end = min(start + frame_len, len(clean))
        out[start:end] += unvoiced[: end - start]

    out = sosfilt(hp_sos, out).astype(np.float32)
    # Mild blend (80% unvoiced breath turbulence + 20% attenuated HP speech) preserves crisp consonant transients
    hp_orig = sosfilt(hp_sos, clean).astype(np.float32)
    blended = 0.80 * _scale_to_dbfs(out, -25.0) + 0.20 * _scale_to_dbfs(hp_orig, -25.0)
    return blended.astype(np.float32)


def _decode_hf_audio(audio_dict: dict[str, Any]) -> tuple[np.ndarray, int]:
    raw_bytes = audio_dict.get("bytes")
    if raw_bytes is not None:
        arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    else:
        arr, sr = sf.read(audio_dict["path"], dtype="float32")
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    if sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    return arr.astype(np.float32), TARGET_SR


def main() -> None:
    rng = np.random.default_rng(42)
    for sub in ("normal", "soft_whisper", "pseudo_whisper", "silence_noise", "hallucination_probes"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)

    print("Loading hf-internal-testing/librispeech_asr_dummy (clean/validation)...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").cast_column(
        "audio", Audio(decode=False)
    )

    speech_items = []
    # Use up to 40 diverse utterances so multi-model sweep completes reliably in a few minutes
    for idx, row in enumerate(ds):
        if idx >= 40:
            break
        audio, _ = _decode_hf_audio(row["audio"])
        norm = _scale_to_dbfs(audio, -20.0)

        room_noise = _pinkish_noise(len(audio), rng) * (10.0 ** (-62.0 / 20.0))
        soft = np.clip(_scale_to_dbfs(audio, -48.0) + room_noise, -0.98, 0.98).astype(np.float32)

        pseudo_raw = _synthesize_pseudo_whisper(norm, TARGET_SR, rng)
        pseudo = np.clip(_scale_to_dbfs(pseudo_raw, -48.0) + room_noise, -0.98, 0.98).astype(np.float32)

        clip_id = f"ls_{idx:03d}"
        p_norm = DATA_DIR / "normal" / f"{clip_id}.wav"
        p_soft = DATA_DIR / "soft_whisper" / f"{clip_id}.wav"
        p_pseudo = DATA_DIR / "pseudo_whisper" / f"{clip_id}.wav"
        sf.write(p_norm, norm, TARGET_SR, subtype="PCM_16")
        sf.write(p_soft, soft, TARGET_SR, subtype="PCM_16")
        sf.write(p_pseudo, pseudo, TARGET_SR, subtype="PCM_16")

        speech_items.append(
            {
                "id": clip_id,
                "speaker_id": str(row.get("speaker_id", "")),
                "duration_sec": round(len(audio) / TARGET_SR, 3),
                "ground_truth": row["text"],
                "normal_path": str(p_norm),
                "soft_whisper_path": str(p_soft),
                "pseudo_whisper_path": str(p_pseudo),
            }
        )

    print(f"Prepared {len(speech_items)} LibriSpeech English samples across 3 acoustic tracks.")

    # Non-speech ambient room + unvoiced breath-only probes to measure false hallucination rate
    silence_items = []
    for idx in range(15):
        dur = 2.0 + (idx % 3) * 0.5
        n = int(dur * TARGET_SR)
        ambient = _pinkish_noise(n, rng) * (10.0 ** (-58.0 / 20.0))
        if idx % 2 == 1:
            # Add a gentle breath exhalation burst in the middle (pure non-lexical breath)
            burst_len = int(0.45 * TARGET_SR)
            start = n // 3
            env = np.hanning(burst_len).astype(np.float32)
            breath = _pinkish_noise(burst_len, rng) * env * (10.0 ** (-44.0 / 20.0))
            ambient[start : start + burst_len] += breath
        ambient = np.clip(ambient, -0.98, 0.98).astype(np.float32)
        p_sil = DATA_DIR / "silence_noise" / f"silence_{idx:03d}.wav"
        sf.write(p_sil, ambient, TARGET_SR, subtype="PCM_16")
        silence_items.append({"id": f"silence_{idx:03d}", "duration_sec": dur, "ground_truth": "", "path": str(p_sil)})

    print("Loading Scicom-intl/Whisper-Hallucination test probes (with tail silence)...")
    ds_h = (
        load_dataset("Scicom-intl/Whisper-Hallucination", split="test", streaming=True)
        .cast_column("audio", Audio(decode=False))
    )
    hallucination_items = []
    for row in ds_h:
        if float(row.get("tail_silence_s", 0.0)) >= 2.0 or len(hallucination_items) < 20:
            if len(hallucination_items) >= 20:
                break
            audio, _ = _decode_hf_audio(row["audio"])
            clip_id = row["id"]
            p_h = DATA_DIR / "hallucination_probes" / f"{clip_id}.wav"
            sf.write(p_h, audio, TARGET_SR, subtype="PCM_16")
            hallucination_items.append(
                {
                    "id": clip_id,
                    "duration_sec": round(len(audio) / TARGET_SR, 3),
                    "tail_silence_s": float(row.get("tail_silence_s", 0.0)),
                    "reference_text": row["reference_text"],
                    "path": str(p_h),
                }
            )

    manifest = {
        "speech_count": len(speech_items),
        "silence_count": len(silence_items),
        "hallucination_probe_count": len(hallucination_items),
        "speech_items": speech_items,
        "silence_items": silence_items,
        "hallucination_items": hallucination_items,
    }
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
