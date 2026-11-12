"""Run isolated multi-model whisper & ASR benchmarks for Hush."""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jiwer
import numpy as np
import soundfile as sf

# Ensure editable/local sottovox_engine is importable even if run outside engine/.
ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from sottovox_engine.asr import ASRConfig, Transcriber, resolve_local_snapshot  # noqa: E402
from sottovox_engine.dsp import DSPConfig, process_audio  # noqa: E402
from sottovox_engine.rule_cleanup import clean_transcript_rules  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

DATASETS_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = DATASETS_DIR / "data" / "manifest.json"
RESULTS_JSON_PATH = DATASETS_DIR / "results.json"
REPORT_MD_PATH = DATASETS_DIR / "REPORT.md"
WORKER_CACHE_DIR = DATASETS_DIR / ".worker_cache"

TEXT_NORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    repo_id: str


MODELS: list[ModelSpec] = [
    ModelSpec("mlx_whisper_large_v3", "whisper-large-v3-mlx", "mlx-whisper", "mlx-community/whisper-large-v3-mlx"),
    ModelSpec("mlx_whisper_turbo", "whisper-large-v3-turbo", "mlx-whisper", "mlx-community/whisper-large-v3-turbo"),
    ModelSpec("mlx_whisper_small", "whisper-small-mlx", "mlx-whisper", "mlx-community/whisper-small-mlx"),
    ModelSpec("mlx_whisper_tiny", "whisper-tiny-mlx", "mlx-whisper", "mlx-community/whisper-tiny-mlx"),
    ModelSpec("faster_whisper_base", "faster-whisper-base (int8 CPU)", "faster-whisper", "Systran/faster-whisper-base"),
    ModelSpec("parakeet_mlx_v3", "parakeet-tdt-0.6b-v3 (MLX)", "parakeet-mlx", "mlx-community/parakeet-tdt-0.6b-v3"),
    ModelSpec(
        "sherpa_parakeet_int8",
        "sherpa-onnx parakeet-tdt-0.6b-v3-int8 (CPU)",
        "sherpa-onnx",
        "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    ),
]


def _to_mb(value: str, unit: str) -> float:
    num = float(value)
    u = unit.upper()
    if u == "GB":
        return num * 1024.0
    if u == "MB":
        return num
    if u == "KB":
        return num / 1024.0
    return num / (1024.0 * 1024.0)


def measure_process_memory() -> dict[str, float]:
    """Measure RSS and macOS `footprint` (includes Metal IOAccelerator unified memory)."""
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    footprint_mb = 0.0
    gpu_ioaccel_mb = 0.0
    proc = subprocess.run(["footprint", str(os.getpid())], capture_output=True, text=True)
    if proc.returncode == 0:
        m_fp = re.search(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB)", proc.stdout)
        if m_fp:
            footprint_mb = _to_mb(m_fp.group(1), m_fp.group(2))
        m_gpu = re.search(r"^\s*([\d.]+)\s*(B|KB|MB|GB)\s+.*IOAccelerator", proc.stdout, re.MULTILINE)
        if m_gpu:
            gpu_ioaccel_mb = _to_mb(m_gpu.group(1), m_gpu.group(2))
    return {
        "ru_maxrss_mb": round(rss_mb, 1),
        "footprint_mb": round(footprint_mb, 1),
        "gpu_ioaccel_mb": round(gpu_ioaccel_mb, 1),
    }


def _resolve_any_snapshot(repo_id: str) -> str:
    """Resolve any local HF repo snapshot (including ONNX/NeMo repos without config.json)."""
    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return resolve_local_snapshot(repo_id)


def snapshot_disk_mb(repo_id: str) -> float:
    resolved = Path(_resolve_any_snapshot(repo_id))
    if not resolved.exists():
        return 0.0
    total = 0
    for item in resolved.rglob("*"):
        if item.is_file():
            # Follow HF snapshot symlinks to actual blob size, counting each file once.
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return round(total / (1024.0 * 1024.0), 1)


def _normalize(text: str) -> str:
    return TEXT_NORM(text or "")


def _compute_wer(refs: list[str], hyps: list[str]) -> float:
    norm_refs = [_normalize(r) or "<empty>" for r in refs]
    norm_hyps = [_normalize(h) for h in hyps]
    return round(float(jiwer.wer(norm_refs, norm_hyps)) * 100.0, 2)


def _make_runner(spec: ModelSpec) -> tuple[Callable[[np.ndarray, int], str], float]:
    """Load model cleanly and return `(transcribe_fn, load_time_ms)`."""
    t0 = time.perf_counter()
    resolved = _resolve_any_snapshot(spec.repo_id)

    if spec.family in ("mlx-whisper", "faster-whisper"):
        transcriber = Transcriber()
        transcriber.warm(engine=spec.family, model=spec.repo_id)  # type: ignore[arg-type]
        load_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        def run_whisper(audio_16k: np.ndarray, _sr: int) -> str:
            cfg = ASRConfig(engine=spec.family, model=spec.repo_id, language="en", whisper_mode=True)  # type: ignore[arg-type]
            return transcriber.transcribe(audio_16k, cfg).text

        return run_whisper, load_ms

    if spec.family == "parakeet-mlx":
        import parakeet_mlx

        model = parakeet_mlx.from_pretrained(resolved)
        # One warm-up pass on 1s silence to trigger Metal graph compile.
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, np.zeros(16000, dtype=np.float32), 16000)
            model.transcribe(tmp.name)
        load_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        def run_parakeet_mlx(audio_16k: np.ndarray, sr: int) -> str:
            if audio_16k.size == 0:
                return ""
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, audio_16k, sr, subtype="PCM_16")
                res = model.transcribe(tmp.name)
                return (res.text or "").strip()

        return run_parakeet_mlx, load_ms

    if spec.family == "sherpa-onnx":
        import sherpa_onnx

        snap = Path(resolved)
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(snap / "encoder.int8.onnx"),
            decoder=str(snap / "decoder.int8.onnx"),
            joiner=str(snap / "joiner.int8.onnx"),
            tokens=str(snap / "tokens.txt"),
            num_threads=4,
            model_type="nemo_transducer",
        )
        load_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        def run_sherpa(audio_16k: np.ndarray, sr: int) -> str:
            if audio_16k.size == 0:
                return ""
            stream = rec.create_stream()
            stream.accept_waveform(sr, audio_16k)
            rec.decode_stream(stream)
            return (stream.result.text or "").strip()

        return run_sherpa, load_ms

    raise ValueError(f"Unknown model family: {spec.family}")


def _prep_audio(path: Path, dsp_enabled: bool) -> tuple[np.ndarray, int]:
    raw_bytes = path.read_bytes()
    res = process_audio(raw_bytes, DSPConfig(enabled=dsp_enabled))
    return res.audio, res.sample_rate


def _eval_speech_condition(
    items: list[dict[str, Any]],
    path_key: str,
    dsp_enabled: bool,
    runner: Callable[[np.ndarray, int], str],
) -> dict[str, Any]:
    refs: list[str] = []
    raw_hyps: list[str] = []
    clean_hyps: list[str] = []
    latencies_ms: list[float] = []
    cleanup_latencies_us: list[float] = []

    for item in items:
        refs.append(item["ground_truth"])
        audio_16k, sr = _prep_audio(Path(item[path_key]), dsp_enabled=dsp_enabled)
        t0 = time.perf_counter()
        hyp = runner(audio_16k, sr)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)
        raw_hyps.append(hyp)

        tc0 = time.perf_counter()
        cleaned = clean_transcript_rules(hyp)
        cleanup_latencies_us.append((time.perf_counter() - tc0) * 1_000_000.0)
        clean_hyps.append(cleaned)

    sorted_lats = sorted(latencies_ms)
    p95_idx = min(len(sorted_lats) - 1, max(0, int(0.95 * len(sorted_lats)) - 1))
    return {
        "n_clips": len(items),
        "dsp_enabled": dsp_enabled,
        "wer_pct": _compute_wer(refs, raw_hyps),
        "wer_after_rules_pct": _compute_wer(refs, clean_hyps),
        "median_latency_ms": round(statistics.median(latencies_ms), 1),
        "p95_latency_ms": round(sorted_lats[p95_idx], 1),
        "mean_cleanup_us": round(statistics.mean(cleanup_latencies_us), 2),
        "sample_outputs": [
            {"id": items[i]["id"], "ref": refs[i], "hyp": raw_hyps[i], "clean_hyp": clean_hyps[i]}
            for i in range(min(3, len(items)))
        ],
    }


def _eval_silence(
    items: list[dict[str, Any]],
    dsp_enabled: bool,
    runner: Callable[[np.ndarray, int], str],
) -> dict[str, Any]:
    hallucinations: list[dict[str, str]] = []
    latencies_ms: list[float] = []
    for item in items:
        audio_16k, sr = _prep_audio(Path(item["path"]), dsp_enabled=dsp_enabled)
        t0 = time.perf_counter()
        hyp = runner(audio_16k, sr)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if _normalize(hyp):
            hallucinations.append({"id": item["id"], "text": hyp})
    total = len(items)
    return {
        "n_clips": total,
        "dsp_enabled": dsp_enabled,
        "hallucination_count": len(hallucinations),
        "hallucination_rate_pct": round((len(hallucinations) / max(1, total)) * 100.0, 1),
        "median_latency_ms": round(statistics.median(latencies_ms), 1),
        "examples": hallucinations[:3],
    }


def _eval_hallucination_probes(
    items: list[dict[str, Any]],
    runner: Callable[[np.ndarray, int], str],
) -> dict[str, Any]:
    refs: list[str] = []
    hyps: list[str] = []
    over_gen_words: list[int] = []
    latencies_ms: list[float] = []
    for item in items:
        ref = item["reference_text"]
        refs.append(ref)
        audio_16k, sr = _prep_audio(Path(item["path"]), dsp_enabled=True)
        t0 = time.perf_counter()
        hyp = runner(audio_16k, sr)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        hyps.append(hyp)
        ref_w = len(_normalize(ref).split())
        hyp_w = len(_normalize(hyp).split())
        over_gen_words.append(max(0, hyp_w - ref_w))
    extra_clips = sum(1 for x in over_gen_words if x >= 3)
    return {
        "n_clips": len(items),
        "wer_pct": _compute_wer(refs, hyps),
        "extra_word_clips_ge3": extra_clips,
        "extra_word_rate_pct": round((extra_clips / max(1, len(items))) * 100.0, 1),
        "mean_extra_words": round(statistics.mean(over_gen_words), 2),
        "median_latency_ms": round(statistics.median(latencies_ms), 1),
    }


def run_worker(spec_key: str) -> dict[str, Any]:
    spec = next(m for m in MODELS if m.key == spec_key)
    manifest = json.loads(MANIFEST_PATH.read_text())
    speech_items = manifest["speech_items"]
    silence_items = manifest["silence_items"]
    hall_items = manifest["hallucination_items"]

    runner, load_ms = _make_runner(spec)
    mem_idle = measure_process_memory()

    conditions = {
        "normal_dsp_off": _eval_speech_condition(speech_items, "normal_path", False, runner),
        "soft_whisper_dsp_off": _eval_speech_condition(speech_items, "soft_whisper_path", False, runner),
        "soft_whisper_dsp_on": _eval_speech_condition(speech_items, "soft_whisper_path", True, runner),
        "pseudo_whisper_dsp_off": _eval_speech_condition(speech_items, "pseudo_whisper_path", False, runner),
        "pseudo_whisper_dsp_on": _eval_speech_condition(speech_items, "pseudo_whisper_path", True, runner),
    }
    silence = {
        "dsp_off": _eval_silence(silence_items, False, runner),
        "dsp_on": _eval_silence(silence_items, True, runner),
    }
    hallucination_probes = _eval_hallucination_probes(hall_items, runner)
    mem_peak = measure_process_memory()

    return {
        "key": spec.key,
        "label": spec.label,
        "family": spec.family,
        "repo_id": spec.repo_id,
        "disk_mb": snapshot_disk_mb(spec.repo_id),
        "load_time_ms": load_ms,
        "memory_idle": mem_idle,
        "memory_peak": mem_peak,
        "conditions": conditions,
        "silence": silence,
        "hallucination_probes": hallucination_probes,
    }


def build_report(results: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Hush Multi-Model Whisper & ASR Benchmark Report",
        "",
        f"- Generated at: `{results['generated_at']}`",
        f"- Dataset manifest: `{MANIFEST_PATH}`",
        f"- Dataset composition: `{results['dataset_summary']['speech_count']}` LibriSpeech clean-validation utterances × 3 conditions (`normal`, `soft_whisper`, `pseudo_whisper`), `{results['dataset_summary']['silence_count']}` ambient silence/breath noise probes, and `{results['dataset_summary']['hallucination_probe_count']}` `Scicom-intl/Whisper-Hallucination` tail-silence probes.",
        "",
        "## Model Comparison Summary",
        "",
        "| Model | Disk (MB) | Peak Footprint (MB) | GPU IOAccel (MB) | Normal WER (%) | Soft Whisper WER Off -> On (%) | Pseudo Whisper WER Off -> On (%) | Hallucination Probe WER (%) | Silence False-Pos On (%) | Soft Whisper Median Latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for m in results["models"]:
        c = m["conditions"]
        mem = m["memory_peak"]
        sw_off = c["soft_whisper_dsp_off"]["wer_pct"]
        sw_on = c["soft_whisper_dsp_on"]["wer_pct"]
        pw_off = c["pseudo_whisper_dsp_off"]["wer_pct"]
        pw_on = c["pseudo_whisper_dsp_on"]["wer_pct"]
        sil_on = m["silence"]["dsp_on"]["hallucination_rate_pct"]
        hp_wer = m["hallucination_probes"]["wer_pct"]
        lat = c["soft_whisper_dsp_on"]["median_latency_ms"]
        lines.append(
            f"| {m['label']} | {m['disk_mb']} | {mem['footprint_mb']} | {mem['gpu_ioaccel_mb']} | "
            f"{c['normal_dsp_off']['wer_pct']} | {sw_off} -> {sw_on} | {pw_off} -> {pw_on} | "
            f"{hp_wer} | {sil_on} | {lat} |"
        )

    lines.extend(
        [
            "",
            "## DSP & Rule Cleanup Impact",
            "",
            "| Model | Soft Whisper DSP Off -> On | Pseudo Whisper DSP Off -> On | Silence False-Pos DSP Off -> On (%) | Rule Cleanup Mean Overhead (us) |",
            "|---|---|---|---:|---:|",
        ]
    )
    for m in results["models"]:
        c = m["conditions"]
        s = m["silence"]
        sw_delta = round(c["soft_whisper_dsp_off"]["wer_pct"] - c["soft_whisper_dsp_on"]["wer_pct"], 2)
        pw_delta = round(c["pseudo_whisper_dsp_off"]["wer_pct"] - c["pseudo_whisper_dsp_on"]["wer_pct"], 2)
        lines.append(
            f"| {m['label']} | {c['soft_whisper_dsp_off']['wer_pct']}% -> {c['soft_whisper_dsp_on']['wer_pct']}% (Δ {sw_delta:+} pp) | "
            f"{c['pseudo_whisper_dsp_off']['wer_pct']}% -> {c['pseudo_whisper_dsp_on']['wer_pct']}% (Δ {pw_delta:+} pp) | "
            f"{s['dsp_off']['hallucination_rate_pct']}% -> {s['dsp_on']['hallucination_rate_pct']}% | "
            f"{c['soft_whisper_dsp_on']['mean_cleanup_us']} |"
        )

    lines.extend(
        [
            "",
            "## Methodology & Honesty Notes",
            "",
            "- Each model runs in a fresh isolated subprocess so `ru_maxrss`, macOS `footprint`, and `IOAccelerator` unified GPU memory reflect true standalone usage.",
            "- `soft_whisper` (`-48 dBFS` RMS + `-62 dBFS` pink noise) and `pseudo_whisper` (LPC unvoiced breath excitation + 240 Hz HPF + `-48 dBFS`) are deterministic synthetic approximations derived from LibriSpeech English ground-truth recordings. They validate DSP gain/filter impact and cross-model ranking cleanly, while real human whispered corpora (e.g., LDC-gated wTIMIT/CHAINS) remain the gold standard for real-world whisper accuracy claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", help="Run a single model worker and emit JSON to stdout.")
    parser.add_argument("--fresh", action="store_true", help="Ignore cached per-worker JSON files.")
    args = parser.parse_args()

    if args.worker:
        payload = run_worker(args.worker)
        print("__JSON_START__")
        print(json.dumps(payload))
        return

    WORKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text())
    model_results: list[dict[str, Any]] = []
    for spec in MODELS:
        cache_file = WORKER_CACHE_DIR / f"{spec.key}.json"
        if cache_file.exists() and not args.fresh:
            print(f"[benchmark] Reusing cached {spec.label} ({cache_file.name})...", flush=True)
            res = json.loads(cache_file.read_text())
        else:
            print(f"[benchmark] Running {spec.label} ({spec.repo_id})...", flush=True)
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", spec.key],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Worker {spec.key} failed:\n{proc.stderr}\n{proc.stdout}")
            marker = "__JSON_START__"
            idx = proc.stdout.rfind(marker)
            if idx == -1:
                raise RuntimeError(f"Missing JSON marker from {spec.key}:\n{proc.stdout}")
            res = json.loads(proc.stdout[idx + len(marker) :].strip())
            cache_file.write_text(json.dumps(res, indent=2))
        model_results.append(res)
        print(
            f"  done: footprint={res['memory_peak']['footprint_mb']} MB, "
            f"normal={res['conditions']['normal_dsp_off']['wer_pct']}%, "
            f"soft_off={res['conditions']['soft_whisper_dsp_off']['wer_pct']}% -> "
            f"soft_on={res['conditions']['soft_whisper_dsp_on']['wer_pct']}%, "
            f"pseudo_on={res['conditions']['pseudo_whisper_dsp_on']['wer_pct']}%, "
            f"silence_fp_on={res['silence']['dsp_on']['hallucination_rate_pct']}%",
            flush=True,
        )

    final = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_summary": {
            "speech_count": manifest["speech_count"],
            "silence_count": manifest["silence_count"],
            "hallucination_probe_count": manifest["hallucination_probe_count"],
        },
        "models": model_results,
    }
    RESULTS_JSON_PATH.write_text(json.dumps(final, indent=2))
    REPORT_MD_PATH.write_text(build_report(final))
    print(f"[benchmark] Wrote {RESULTS_JSON_PATH} and {REPORT_MD_PATH}", flush=True)


if __name__ == "__main__":
    main()
