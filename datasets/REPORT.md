# Hush Multi-Model Whisper & ASR Benchmark Report

- Generated at: `2026-11-12T08:59:45+0100`
- Dataset manifest: `/Volumes/aditya/hush/datasets/data/manifest.json`
- Dataset composition: `40` LibriSpeech clean-validation utterances × 3 conditions (`normal`, `soft_whisper`, `pseudo_whisper`), `15` ambient silence/breath noise probes, and `20` `Scicom-intl/Whisper-Hallucination` tail-silence probes.

## Model Comparison Summary

| Model | Disk (MB) | Peak Footprint (MB) | GPU IOAccel (MB) | Normal WER (%) | Soft Whisper WER Off -> On (%) | Pseudo Whisper WER Off -> On (%) | Hallucination Probe WER (%) | Silence False-Pos On (%) | Soft Whisper Median Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| whisper-large-v3-mlx | 2940.7 | 6563.0 | 5919.0 | 4.57 | 6.64 -> 5.75 | 11.65 -> 12.39 | 100.0 | 100.0 | 625.3 |
| whisper-large-v3-turbo | 1539.2 | 3810.0 | 3491.0 | 5.01 | 5.6 -> 6.05 | 12.68 -> 12.54 | 104.41 | 100.0 | 387.0 |
| whisper-small-mlx | 459.0 | 2990.0 | 2500.0 | 8.26 | 9.59 -> 10.47 | 24.19 -> 24.63 | 1389.71 | 100.0 | 141.7 |
| whisper-tiny-mlx | 71.0 | 1638.0 | 1253.0 | 12.54 | 20.65 -> 19.91 | 72.57 -> 67.7 | 1069.12 | 13.3 | 30.4 |
| faster-whisper-base (int8 CPU) | 141.0 | 552.0 | 0.1 | 8.55 | 13.86 -> 14.9 | 33.04 -> 30.83 | 104.41 | 86.7 | 592.2 |
| parakeet-tdt-0.6b-v3 (MLX) | 2392.8 | 9599.0 | 9252.0 | 3.24 | 5.16 -> 3.83 | 11.65 -> 8.85 | 95.59 | 46.7 | 119.4 |
| sherpa-onnx parakeet-tdt-0.6b-v3-int8 (CPU) | 640.1 | 1893.0 | 0.1 | 3.83 | 5.9 -> 7.08 | 24.04 -> 22.42 | 100.0 | 46.7 | 241.4 |

## DSP & Rule Cleanup Impact

| Model | Soft Whisper DSP Off -> On | Pseudo Whisper DSP Off -> On | Silence False-Pos DSP Off -> On (%) | Rule Cleanup Mean Overhead (us) |
|---|---|---|---:|---:|
| whisper-large-v3-mlx | 6.64% -> 5.75% (Δ +0.89 pp) | 11.65% -> 12.39% (Δ -0.74 pp) | 100.0% -> 100.0% | 74.83 |
| whisper-large-v3-turbo | 5.6% -> 6.05% (Δ -0.45 pp) | 12.68% -> 12.54% (Δ +0.14 pp) | 100.0% -> 100.0% | 127.47 |
| whisper-small-mlx | 9.59% -> 10.47% (Δ -0.88 pp) | 24.19% -> 24.63% (Δ -0.44 pp) | 100.0% -> 100.0% | 59.01 |
| whisper-tiny-mlx | 20.65% -> 19.91% (Δ +0.74 pp) | 72.57% -> 67.7% (Δ +4.87 pp) | 20.0% -> 13.3% | 32.78 |
| faster-whisper-base (int8 CPU) | 13.86% -> 14.9% (Δ -1.04 pp) | 33.04% -> 30.83% (Δ +2.21 pp) | 100.0% -> 86.7% | 51.96 |
| parakeet-tdt-0.6b-v3 (MLX) | 5.16% -> 3.83% (Δ +1.33 pp) | 11.65% -> 8.85% (Δ +2.8 pp) | 0.0% -> 46.7% | 81.22 |
| sherpa-onnx parakeet-tdt-0.6b-v3-int8 (CPU) | 5.9% -> 7.08% (Δ -1.18 pp) | 24.04% -> 22.42% (Δ +1.62 pp) | 0.0% -> 46.7% | 70.84 |

## Methodology & Honesty Notes

- Each model runs in a fresh isolated subprocess so `ru_maxrss`, macOS `footprint`, and `IOAccelerator` unified GPU memory reflect true standalone usage.
- `soft_whisper` (`-48 dBFS` RMS + `-62 dBFS` pink noise) and `pseudo_whisper` (LPC unvoiced breath excitation + 240 Hz HPF + `-48 dBFS`) are deterministic synthetic approximations derived from LibriSpeech English ground-truth recordings. They validate DSP gain/filter impact and cross-model ranking cleanly, while real human whispered corpora (e.g., LDC-gated wTIMIT/CHAINS) remain the gold standard for real-world whisper accuracy claims.
