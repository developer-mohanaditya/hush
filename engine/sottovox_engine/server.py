from __future__ import annotations

import argparse
import asyncio
import base64
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

from .asr import (
    DEFAULT_ENGINE,
    DEFAULT_FASTER_MODEL,
    DEFAULT_INITIAL_PROMPT,
    DEFAULT_MLX_MODEL,
    DEFAULT_PARAKEET_INT8_MODEL,
    ASRConfig,
    EngineKind,
    Transcriber,
    available_models,
)
from .cleanup import (
    clean_transcript,
    clean_transcript_custom_openai,
    list_custom_models,
    list_ollama_models,
    pick_default_model,
)
from .dsp import DSPConfig, process_audio
from .rule_cleanup import clean_transcript_rules


def _default_model_for(engine: EngineKind) -> str:
    if engine == "sherpa-parakeet":
        return DEFAULT_PARAKEET_INT8_MODEL
    if engine == "mlx-whisper":
        return DEFAULT_MLX_MODEL
    return DEFAULT_FASTER_MODEL


def create_app(warm_on_start: bool = False) -> FastAPI:
    transcriber = Transcriber()
    inference_lock = asyncio.Lock()
    state = {"warmed": False, "warm_error": None}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if warm_on_start:
            try:
                await asyncio.to_thread(transcriber.warm, DEFAULT_ENGINE, DEFAULT_PARAKEET_INT8_MODEL)
                state["warmed"] = True
            except Exception as exc:
                state["warm_error"] = str(exc)
        yield

    app = FastAPI(title="Hush Engine", version="0.2.0", lifespan=lifespan)
    app.state.transcriber = transcriber

    @app.get("/health")
    async def health() -> dict:
        ollama_models = await list_ollama_models()
        return {
            "ok": True,
            "warmed": state["warmed"],
            "warm_error": state["warm_error"],
            "default_engine": DEFAULT_ENGINE,
            "default_model": DEFAULT_PARAKEET_INT8_MODEL,
            "asr_models": available_models(),
            "ollama_models": ollama_models,
            "default_ollama_model": pick_default_model(ollama_models),
        }

    @app.post("/warm")
    async def warm(
        engine: EngineKind = Form(DEFAULT_ENGINE),
        model: str | None = Form(None),
    ) -> dict:
        t0 = time.perf_counter()
        async with inference_lock:
            resolved = await asyncio.to_thread(transcriber.warm, engine, model)
            state["warmed"] = True
            state["warm_error"] = None
        return {
            "ok": True,
            "engine": engine,
            "resolved": resolved,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    @app.post("/llm/models")
    async def fetch_llm_models(
        base_url: str = Form(...),
        api_key: str = Form(""),
    ) -> dict:
        models, err = await list_custom_models(base_url=base_url, api_key=api_key or None)
        return {"ok": err is None, "models": models, "error": err}

    @app.post("/transcribe")
    async def transcribe_endpoint(
        audio: UploadFile = File(...),
        dsp_enabled: bool = Form(True),
        highpass_hz: float = Form(95.0),
        target_rms_dbfs: float = Form(-18.0),
        max_gain_db: float = Form(32.0),
        silence_gate_dbfs: float = Form(-65.0),
        trim_silence: bool = Form(True),
        engine: EngineKind = Form(DEFAULT_ENGINE),
        model: str | None = Form(None),
        language: str = Form("en"),
        whisper_mode: bool = Form(True),
        initial_prompt: str = Form(DEFAULT_INITIAL_PROMPT),
        cleanup: bool = Form(True),
        cleanup_mode: str = Form("rules"),
        ollama_model: str | None = Form(None),
        custom_base_url: str = Form(""),
        custom_api_key: str = Form(""),
        custom_model: str = Form(""),
        return_wav: bool = Form(True),
    ) -> dict:
        raw_bytes = await audio.read()

        t_dsp = time.perf_counter()
        dsp_cfg = DSPConfig(
            enabled=dsp_enabled,
            highpass_hz=highpass_hz,
            target_rms_dbfs=target_rms_dbfs,
            max_gain_db=max_gain_db,
            silence_gate_dbfs=silence_gate_dbfs,
            trim_silence=trim_silence,
        )
        dsp_res = await asyncio.to_thread(process_audio, raw_bytes, dsp_cfg)
        dsp_ms = (time.perf_counter() - t_dsp) * 1000.0

        chosen_model = model or _default_model_for(engine)
        asr_cfg = ASRConfig(
            engine=engine,
            model=chosen_model,
            language=language,
            whisper_mode=whisper_mode,
            initial_prompt=initial_prompt.strip() or None,
        )

        if dsp_res.stats.too_quiet or dsp_res.audio.size < 1600:
            return {
                "raw_text": "",
                "clean_text": "",
                "stats": dsp_res.stats.to_dict(),
                "engine": engine,
                "model": chosen_model,
                "language": language,
                "cleanup_applied": False,
                "cleanup_model": None,
                "cleanup_error": None,
                "timings_ms": {"dsp": round(dsp_ms, 1), "asr": 0.0, "cleanup": 0.0},
                "boosted_wav_b64": (
                    base64.b64encode(dsp_res.to_wav_bytes()).decode("ascii")
                    if return_wav and dsp_res.audio.size > 0
                    else None
                ),
            }

        t_asr = time.perf_counter()
        async with inference_lock:
            asr_res = await asyncio.to_thread(transcriber.transcribe, dsp_res.audio, asr_cfg)
            state["warmed"] = True
        asr_ms = (time.perf_counter() - t_asr) * 1000.0

        t_clean = time.perf_counter()
        if cleanup and asr_res.text and cleanup_mode != "none":
            if cleanup_mode == "ollama":
                clean_res = await clean_transcript(asr_res.text, model=ollama_model or None)
            elif cleanup_mode == "custom_openai":
                clean_res = await clean_transcript_custom_openai(
                    asr_res.text,
                    base_url=custom_base_url,
                    api_key=custom_api_key or None,
                    model=custom_model,
                )
            else:
                from .cleanup import CleanupResult

                cleaned = clean_transcript_rules(asr_res.text)
                clean_res = CleanupResult(text=cleaned, applied=True, model="rules (<1 ms)")
        else:
            from .cleanup import CleanupResult

            clean_res = CleanupResult(text=asr_res.text, applied=False, model=None)
        clean_ms = (time.perf_counter() - t_clean) * 1000.0

        boosted_b64 = (
            base64.b64encode(dsp_res.to_wav_bytes()).decode("ascii") if return_wav else None
        )
        return {
            "raw_text": asr_res.text,
            "clean_text": clean_res.text,
            "stats": dsp_res.stats.to_dict(),
            "engine": asr_res.engine,
            "model": asr_res.model,
            "language": asr_res.language,
            "cleanup_applied": clean_res.applied,
            "cleanup_model": clean_res.model,
            "cleanup_error": clean_res.error,
            "timings_ms": {
                "dsp": round(dsp_ms, 1),
                "asr": round(asr_ms, 1),
                "cleanup": round(clean_ms, 1),
            },
            "boosted_wav_b64": boosted_b64,
        }

    return app


app = create_app(warm_on_start=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sottovox local whisper-to-text server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--warm", action="store_true", help="Preload default MLX Whisper model.")
    args = parser.parse_args()
    uvicorn.run(create_app(warm_on_start=args.warm), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
