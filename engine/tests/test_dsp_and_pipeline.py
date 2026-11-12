from __future__ import annotations

import io
import math

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from sottovox_engine.asr import available_models, resolve_local_snapshot
from sottovox_engine.dsp import DSPConfig, process_audio
from sottovox_engine.server import create_app


def _wav_bytes(samples: np.ndarray, sr: int = 16_000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_whisper_dsp_boosts_quiet_signal_and_attenuates_plosives() -> None:
    sr = 48_000
    t = np.linspace(0, 1.4, int(sr * 1.4), endpoint=False)
    # 1.2 kHz quiet whisper-like tone (~ -46 dBFS) + 30 Hz plosive breath thump (~ -24 dBFS).
    quiet_speech = 0.005 * np.sin(2 * math.pi * 1200 * t)
    breath_thump = 0.06 * np.sin(2 * math.pi * 30 * t)
    raw = _wav_bytes(quiet_speech + breath_thump, sr=sr)

    res = process_audio(raw, DSPConfig(enabled=True, highpass_hz=95.0, target_rms_dbfs=-18.0))
    assert res.sample_rate == 16_000
    assert res.stats.applied_gain_db > 15.0
    assert res.stats.processed_rms_dbfs > -24.0
    assert not res.stats.too_quiet
    # Confirm 30 Hz energy was stripped by the high-pass filter.
    spectrum = np.abs(np.fft.rfft(res.audio))
    freqs = np.fft.rfftfreq(res.audio.size, 1 / res.sample_rate)
    low_band = spectrum[(freqs >= 15) & (freqs <= 45)].max()
    speech_band = spectrum[(freqs >= 1100) & (freqs <= 1300)].max()
    assert speech_band > low_band * 10


def test_silent_audio_is_flagged_too_quiet_without_amplifying_noise() -> None:
    sr = 16_000
    silence = np.zeros(sr, dtype=np.float32)
    res = process_audio(_wav_bytes(silence, sr=sr))
    assert res.stats.too_quiet is True
    assert res.stats.applied_gain_db == 0.0


def test_health_and_quiet_transcribe_endpoint() -> None:
    app = create_app(warm_on_start=False)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ok"] is True
    assert any(m["id"] == "mlx-community/whisper-large-v3-turbo" for m in body["asr_models"])

    silence = _wav_bytes(np.zeros(16_000, dtype=np.float32))
    resp = client.post(
        "/transcribe",
        files={"audio": ("silence.wav", silence, "audio/wav")},
        data={"cleanup": "false"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["raw_text"] == ""
    assert data["stats"]["too_quiet"] is True


def test_local_cache_snapshot_discovery() -> None:
    models = available_models()
    assert len(models) >= 2
    full_v3 = next(m for m in models if m["id"] == "mlx-community/whisper-large-v3-mlx")
    turbo = next(m for m in models if m["id"] == "mlx-community/whisper-large-v3-turbo")
    assert full_v3["cached"] is True
    assert turbo["cached"] is True
    assert resolve_local_snapshot("mlx-community/whisper-large-v3-mlx") != "mlx-community/whisper-large-v3-mlx"
    assert resolve_local_snapshot("mlx-community/whisper-large-v3-turbo") != "mlx-community/whisper-large-v3-turbo"


def test_modulation_gate_allows_soft_whisper_and_blocks_flat_hum() -> None:
    sr = 16_000
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    # Stationary noise around -61 dBFS (below -56 dBFS, low crest/modulation ratio) -> should be gated.
    flat_hum = rng.normal(0.0, 10 ** (-61.0 / 20.0), size=t.size).astype(np.float32)
    hum_res = process_audio(_wav_bytes(flat_hum, sr=sr), DSPConfig(silence_gate_dbfs=-65.0))
    assert hum_res.stats.too_quiet is True

    # Modulated soft whisper bursts around -61 dBFS active RMS (high syllable contrast) -> should pass.
    envelope = (0.5 + 0.5 * np.sin(2 * math.pi * 3.0 * t) > 0.85).astype(np.float32)
    whisper_burst = (envelope * 0.0022 * np.sin(2 * math.pi * 1100 * t)).astype(np.float32)
    wh_res = process_audio(_wav_bytes(whisper_burst, sr=sr), DSPConfig(silence_gate_dbfs=-65.0))
    assert wh_res.stats.too_quiet is False
    assert wh_res.stats.applied_gain_db > 0.0
