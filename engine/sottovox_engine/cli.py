from __future__ import annotations

import argparse
import asyncio
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .asr import DEFAULT_FASTER_MODEL, DEFAULT_MLX_MODEL, ASRConfig, Transcriber
from .cleanup import clean_transcript
from .dsp import DSPConfig, process_audio


def list_devices() -> None:
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            default_mark = "*" if idx == sd.default.device[0] else " "
            print(f"  [{idx}] {default_mark} {dev['name']} ({int(dev['default_samplerate'])} Hz)")


def record_until_enter(device: int | None) -> bytes:
    frames: list[np.ndarray] = []
    samplerate = 16_000
    if device is not None:
        info = sd.query_devices(device)
        samplerate = int(info["default_samplerate"]) or 16_000

    def callback(indata, _frame_count, _time_info, _status) -> None:
        frames.append(indata.copy())

    input("Press ENTER to start whispering...")
    with sd.InputStream(device=device, channels=1, samplerate=samplerate, dtype="float32", callback=callback):
        input("Whispering... press ENTER to stop.")

    if not frames:
        return b""
    audio = np.concatenate(frames, axis=0).squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def run_once(
    raw_bytes: bytes,
    engine: str,
    model: str | None,
    no_dsp: bool,
    no_cleanup: bool,
    ollama_model: str | None,
    copy: bool,
) -> int:
    dsp_res = process_audio(raw_bytes, DSPConfig(enabled=not no_dsp))
    stats = dsp_res.stats
    print(
        f"[DSP] dur={stats.duration_sec}s -> {stats.trimmed_duration_sec}s | "
        f"raw={stats.raw_rms_dbfs} dBFS (peak {stats.raw_peak_dbfs}) | "
        f"gain=+{stats.applied_gain_db} dB -> {stats.processed_rms_dbfs} dBFS",
        file=sys.stderr,
    )
    if stats.too_quiet:
        print("[!] Audio was essentially silent (< -68 dBFS active). Check input device or mic mute.", file=sys.stderr)
        return 1

    transcriber = Transcriber()
    chosen = model or (DEFAULT_MLX_MODEL if engine == "mlx-whisper" else DEFAULT_FASTER_MODEL)
    asr_res = transcriber.transcribe(dsp_res.audio, ASRConfig(engine=engine, model=chosen))  # type: ignore[arg-type]
    print(f"[RAW] {asr_res.text}")

    final_text = asr_res.text
    if not no_cleanup and asr_res.text:
        clean_res = await clean_transcript(asr_res.text, model=ollama_model)
        if clean_res.applied:
            final_text = clean_res.text
            print(f"[CLEAN · {clean_res.model}] {final_text}")
        elif clean_res.error:
            print(f"[cleanup skipped: {clean_res.error}]", file=sys.stderr)

    if copy and final_text:
        subprocess.run(["pbcopy"], input=final_text.encode("utf-8"), check=False)
        print("[copied to clipboard]", file=sys.stderr)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sottovox CLI for testing library whispers.")
    parser.add_argument("--file", type=Path, help="Transcribe an existing WAV/CAF/M4A file.")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices.")
    parser.add_argument("--device", type=int, help="Input device index for live capture.")
    parser.add_argument("--engine", choices=["mlx-whisper", "faster-whisper"], default="mlx-whisper")
    parser.add_argument("--model", help="Override ASR repo or local path.")
    parser.add_argument("--ollama-model", help="Ollama model for prompt cleanup.")
    parser.add_argument("--no-dsp", action="store_true", help="Disable whisper high-pass + AGC boost.")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip Ollama prompt cleanup.")
    parser.add_argument("--copy", action="store_true", help="Copy final output via pbcopy.")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.file:
        raw = args.file.read_bytes()
    else:
        raw = record_until_enter(args.device)
        if not raw:
            print("No audio captured.", file=sys.stderr)
            sys.exit(1)

    code = asyncio.run(
        run_once(
            raw_bytes=raw,
            engine=args.engine,
            model=args.model,
            no_dsp=args.no_dsp,
            no_cleanup=args.no_cleanup,
            ollama_model=args.ollama_model,
            copy=args.copy,
        )
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
