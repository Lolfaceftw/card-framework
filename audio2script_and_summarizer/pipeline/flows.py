"""High-level pipeline execution flows extracted from orchestrator."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, ContextManager, Protocol


class _RichMessagePrinter(Protocol):
    """Callable contract for rich/plain message printers."""

    def __call__(self, message: str, /, use_rich: bool) -> None:
        """Emit one message."""


class _TranscriptPrompt(Protocol):
    """Callable contract for transcript selection prompts."""

    def __call__(self, search_root: Path, use_rich: bool) -> Path:
        """Return selected transcript JSON path."""


class _SpeakerWpmFormatter(Protocol):
    """Callable contract for per-speaker WPM formatting."""

    def __call__(
        self, per_speaker_wpm: dict[str, float], max_items: int = 6
    ) -> str:
        """Return formatted summary string."""


class _OutputCaptureContextFactory(Protocol):
    """Callable contract for output-capture context managers."""

    def __call__(self, enabled: bool) -> ContextManager[None]:
        """Return a context manager that captures runtime output."""


def _resolve_stage175_cache_settings(args: Any) -> tuple[str, str]:
    """Resolve Stage 1.75 calibration cache mode and directory from args."""
    cache_mode_raw = str(
        getattr(args, "wpm_calibration_cache_mode", "auto")
    ).strip()
    cache_mode = cache_mode_raw.lower() or "auto"
    cache_dir = str(
        getattr(
            args,
            "wpm_calibration_cache_dir",
            "artifacts/cache/wpm_calibration",
        )
    ).strip()
    return cache_mode, cache_dir


def _compute_tts_preflight_wpm(
    *,
    args: Any,
    voice_dir: str,
    runtime_device: str,
    transcript_json_path: str,
    capture_stage175_runtime_output: bool,
    progress_cb: Callable[[Any], None] | None,
    _capture_stage175_output_lines: _OutputCaptureContextFactory,
    logger: logging.Logger,
) -> tuple[float, dict[str, float], Any, bool, Path | None]:
    """Compute Stage 1.75 WPM from TTS preflight with optional cache reuse.

    Returns:
        Tuple of weighted average WPM, per-speaker WPM map, calibration object,
        cache-hit flag, and cache file path when read/written.
    """
    from audio2script_and_summarizer.tts_pacing_calibration import (
        calibrate_tts_pacing_profiles,
        estimate_weighted_wpm_from_transcript,
    )
    from audio2script_and_summarizer.wpm_calibration_cache import (
        build_calibration_fingerprint,
        cache_path_for_fingerprint,
        load_cached_tts_pacing,
        save_cached_tts_pacing,
    )

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (
        repo_root
        / "voice-cloner-and-interjector"
        / "checkpoints"
        / "config.yaml"
    )
    model_dir = repo_root / "voice-cloner-and-interjector" / "checkpoints"
    cache_mode, cache_dir = _resolve_stage175_cache_settings(args)

    cache_fingerprint: str | None = None
    cache_path: Path | None = None
    cached_calibration: Any | None = None
    if cache_mode != "off":
        try:
            cache_fingerprint = build_calibration_fingerprint(
                voice_dir=voice_dir,
                device=runtime_device,
                cfg_path=str(cfg_path),
                model_dir=str(model_dir),
                presets_path=args.calibration_presets_path,
            )
            cache_path = cache_path_for_fingerprint(
                cache_dir=cache_dir,
                fingerprint=cache_fingerprint,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Stage 1.75 cache fingerprint generation failed; continuing without cache reuse: %s",
                exc,
            )

    if cache_mode == "auto" and cache_fingerprint is not None:
        cached_calibration = load_cached_tts_pacing(
            cache_dir=cache_dir,
            fingerprint=cache_fingerprint,
        )
        if cached_calibration is not None:
            avg_wpm, per_speaker_wpm = estimate_weighted_wpm_from_transcript(
                transcript_json_path=transcript_json_path,
                calibration=cached_calibration,
            )
            return avg_wpm, per_speaker_wpm, cached_calibration, True, cache_path

    with _capture_stage175_output_lines(enabled=capture_stage175_runtime_output):
        _, _, tts_pacing = calibrate_tts_pacing_profiles(
            voice_dir=voice_dir,
            device=runtime_device,
            cfg_path=str(cfg_path),
            model_dir=str(model_dir),
            presets_path=args.calibration_presets_path,
            progress_cb=progress_cb,
        )
    avg_wpm, per_speaker_wpm = estimate_weighted_wpm_from_transcript(
        transcript_json_path=transcript_json_path,
        calibration=tts_pacing,
    )

    if cache_mode in {"auto", "refresh"} and cache_fingerprint is not None:
        try:
            cache_path = save_cached_tts_pacing(
                cache_dir=cache_dir,
                fingerprint=cache_fingerprint,
                calibration=tts_pacing,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 1.75 cache save failed: %s", exc)
    return avg_wpm, per_speaker_wpm, tts_pacing, False, cache_path


def run_pipeline_modes(
    *,
    args: Any,
    use_rich: bool,
    dashboard: Any,
    runtime_device: str,
    normalized_wpm_source: str,
    target_minutes: float | None,
    llm_provider: str | None,
    input_path: str,
    current_env: dict[str, str],
    _ACTIVE_DASHBOARD: Any | None,
    _print_stage_banner: _RichMessagePrinter,
    _print_checkpoint: _RichMessagePrinter,
    _print_error: _RichMessagePrinter,
    _print_warning: _RichMessagePrinter,
    _print_success: _RichMessagePrinter,
    _print_info: _RichMessagePrinter,
    _count_wav_files: Callable[[str], int],
    _prompt_for_transcript_json: _TranscriptPrompt,
    _format_speaker_wpm_summary: _SpeakerWpmFormatter,
    _resolve_deepseek_agent_max_tool_rounds: Callable[..., tuple[int, str]],
    _run_stage_command: Callable[..., None],
    _run_stage3_from_summary: Callable[..., tuple[Path, Path, bool, float]],
    _calculate_corrected_word_budget: Callable[..., int],
    _update_summary_report_duration_metrics: Callable[..., None],
    _capture_stage175_output_lines: _OutputCaptureContextFactory,
    logger: logging.Logger,
) -> int:
    """Execute skip and full pipeline flow branches.

    Returns:
        Process exit code for the selected mode.
    """
    measured_duration_seconds: float | None = None
    final_wav_path: Path | None = None
    interjection_log_path: Path | None = None
    per_speaker_wpm: dict[str, float] = {}
    tts_pacing_calibration: Any | None = None

    if args.skip_a2s_summary:
        _print_stage_banner(
            "[SKIP] STAGE 1/1.5/1.75/2: Using Existing Summary JSON",
            use_rich=use_rich,
        )
        dashboard.complete_stage("Stage 1 skipped (--skip-a2s-summary)")
        dashboard.complete_stage("Stage 1.5 skipped (--skip-a2s-summary)")
        dashboard.complete_stage("Stage 1.75 skipped (--skip-a2s-summary)")
        dashboard.complete_stage("Stage 2 skipped (--skip-a2s-summary)")

        if args.summary_json:
            selected_summary = Path(args.summary_json).resolve()
            if not selected_summary.exists():
                _print_error(
                    f"[ERROR] Summary JSON not found: {selected_summary}",
                    use_rich=use_rich,
                )
                return 1
        else:
            search_root = Path(args.skip_a2s_search_root).resolve()
            from audio2script_and_summarizer.stage3_voice import (
                discover_summary_json_files,
                select_latest_summary_json,
            )

            discovered = discover_summary_json_files(search_root)
            if not discovered:
                _print_error(
                    f"[ERROR] No summary JSON files found under {search_root}.",
                    use_rich=use_rich,
                )
                return 1
            selected_summary = select_latest_summary_json(search_root)
            _print_info(
                f"[INFO] --skip-a2s-summary discovered {len(discovered)} summary candidate(s).",
                use_rich=use_rich,
            )
        _print_success(
            f"[SUCCESS] Selected summary JSON: {selected_summary}",
            use_rich=use_rich,
        )

        _print_stage_banner(
            "[START] STAGE 3: Voice Cloning + Interjections",
            use_rich=use_rich,
        )
        _print_checkpoint(
            "Stage 3: running IndexTTS2 synthesis and interjection overlay.",
            use_rich=use_rich,
        )
        stage3_output_override = (
            Path(args.stage3_output).resolve() if args.stage3_output else None
        )
        try:
            (
                final_wav_path,
                interjection_log_path,
                mistral_enabled,
                measured_duration_seconds,
            ) = _run_stage3_from_summary(
                summary_json_path=selected_summary,
                output_wav_path=stage3_output_override,
                runtime_device=runtime_device,
                interjection_max_ratio=args.interjection_max_ratio,
                mistral_model_id=args.mistral_model_id,
                mistral_max_new_tokens=args.mistral_max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            _print_error(f"[ERROR] Stage 3 failed: {exc}", use_rich=use_rich)
            return 1
        dashboard.complete_stage("Stage 3 complete")
        if not mistral_enabled:
            _print_warning(
                "[WARN] Stage 3 ran without Mistral interjections because the model was unavailable.",
                use_rich=use_rich,
            )
        _print_stage_banner("[DONE] PIPELINE COMPLETE", use_rich=use_rich)
        _print_success(f"Summary input: {selected_summary}", use_rich=use_rich)
        _print_success(f"Final audio: {final_wav_path}", use_rich=use_rich)
        _print_success(
            f"Final duration: {measured_duration_seconds:.2f}s",
            use_rich=use_rich,
        )
        _print_success(
            f"Interjection log: {interjection_log_path}", use_rich=use_rich
        )
        return 0

    if args.skip_a2s:
        _print_stage_banner(
            "[SKIP] STAGE 1/1.5: Audio2Script Bypassed", use_rich=use_rich
        )
        _print_checkpoint(
            "Stage 1 and Stage 1.5 skipped via --skip-a2s.",
            use_rich=use_rich,
        )
        dashboard.complete_stage("Stage 1 skipped (--skip-a2s)")
        dashboard.complete_stage("Stage 1.5 skipped (--skip-a2s)")

        search_root = Path(args.skip_a2s_search_root).resolve()
        _print_checkpoint(
            f"Discovering transcript JSON files under: {search_root}",
            use_rich=use_rich,
        )
        try:
            selected_transcript = _prompt_for_transcript_json(
                search_root=search_root,
                use_rich=use_rich,
            )
        except FileNotFoundError as exc:
            _print_error(f"[ERROR] {exc}", use_rich=use_rich)
            return 1

        diarization_json = str(selected_transcript)
        _print_success(
            f"[SUCCESS] Selected transcript JSON: {diarization_json}",
            use_rich=use_rich,
        )

        base_name = os.path.splitext(diarization_json)[0]
        voice_dir = args.voice_dir or f"{base_name}_voices"
        voice_sample_count = _count_wav_files(voice_dir)
        if voice_sample_count <= 0:
            _print_error(
                f"[ERROR] No speaker samples found in voice dir: {voice_dir}",
                use_rich=use_rich,
            )
            _print_warning(
                "Pass --voice-dir pointing to an existing <audio>_voices directory.",
                use_rich=use_rich,
            )
            return 1

        _print_stage_banner(
            "[START] STAGE 1.75: WPM Calibration",
            use_rich=use_rich,
        )
        if normalized_wpm_source == "tts_preflight":
            _print_checkpoint(
                "Stage 1.75: running emotion-aware TTS preflight pacing calibration.",
                use_rich=use_rich,
            )
        else:
            _print_checkpoint(
                "Stage 1.75: deriving WPM from selected transcript JSON.",
                use_rich=use_rich,
            )

        try:
            if normalized_wpm_source == "tts_preflight":
                capture_stage175_runtime_output = (
                    _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled
                )
                (
                    avg_wpm,
                    per_speaker_wpm,
                    tts_pacing,
                    cache_hit,
                    cache_path,
                ) = _compute_tts_preflight_wpm(
                    args=args,
                    voice_dir=voice_dir,
                    runtime_device=runtime_device,
                    transcript_json_path=diarization_json,
                    capture_stage175_runtime_output=capture_stage175_runtime_output,
                    progress_cb=None,
                    _capture_stage175_output_lines=_capture_stage175_output_lines,
                    logger=logger,
                )
                tts_pacing_calibration = tts_pacing
                if cache_hit and cache_path is not None:
                    _print_info(
                        (
                            "[INFO] Stage 1.75 reused cached TTS preflight "
                            f"calibration: {cache_path}"
                        ),
                        use_rich=use_rich,
                    )
                elif cache_path is not None:
                    _print_info(
                        (
                            "[INFO] Stage 1.75 wrote TTS preflight calibration "
                            f"cache: {cache_path}"
                        ),
                        use_rich=use_rich,
                    )
            else:
                from audio2script_and_summarizer.transcript_wpm import (
                    compute_wpm_from_transcript,
                )

                avg_wpm, per_speaker_wpm = compute_wpm_from_transcript(
                    diarization_json
                )
        except Exception as exc:  # noqa: BLE001
            _print_error(
                f"[ERROR] WPM derivation failed: {exc}",
                use_rich=use_rich,
            )
            return 1
        dashboard.complete_stage("Stage 1.75 complete")

        _print_info(
            f"[INFO] Stage 1.75 source={normalized_wpm_source}; per-speaker WPM: "
            f"{_format_speaker_wpm_summary(per_speaker_wpm)}",
            use_rich=use_rich,
        )
        if target_minutes is None:
            _print_error(
                "[ERROR] target minutes is required for --skip-a2s summarization mode.",
                use_rich=use_rich,
            )
            return 1
        word_budget = max(1, int(round(avg_wpm * target_minutes)))
        _print_success(
            f"[SUCCESS] WPM source={normalized_wpm_source}; calibrated WPM={avg_wpm:.2f}; "
            f"target_minutes={target_minutes:.2f}; word_budget={word_budget}",
            use_rich=use_rich,
        )

        _print_stage_banner("[START] STAGE 2: Summarizer", use_rich=use_rich)
        _print_checkpoint(
            "Stage 2: preparing summarizer subprocess", use_rich=use_rich
        )

        summary_output = f"{base_name}_summary.json"
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            _print_error(
                "[ERROR] No DeepSeek API key found. Set DEEPSEEK_API_KEY.",
                use_rich=use_rich,
            )
            return 1

        target_duration_seconds = target_minutes * 60.0
        current_word_budget = word_budget
        correction_passes_used = 0
        max_correction_passes = args.max_duration_correction_passes
        measured_duration_seconds = None
        final_wav_path = None
        interjection_log_path = None
        final_mistral_enabled = True

        while True:
            pass_label = correction_passes_used + 1
            resolved_max_tool_rounds, rounds_source = (
                _resolve_deepseek_agent_max_tool_rounds(
                    configured_max_tool_rounds=args.deepseek_agent_max_tool_rounds,
                    current_word_budget=current_word_budget,
                    target_minutes=target_minutes,
                )
            )
            _print_info(
                (
                    f"[INFO] Stage 2 pass {pass_label}: "
                    f"DeepSeek max_tool_rounds={resolved_max_tool_rounds} "
                    f"(source={rounds_source}; "
                    f"word_budget={current_word_budget}; "
                    f"target_minutes={target_minutes:.2f})."
                ),
                use_rich=use_rich,
            )
            summarize_cmd = [
                sys.executable,
                "-m",
                "audio2script_and_summarizer.summarizer_deepseek",
                "--transcript",
                diarization_json,
                "--voice-dir",
                voice_dir,
                "--output",
                summary_output,
                "--api-key",
                api_key,
                "--max-completion-tokens",
                str(args.deepseek_max_completion_tokens),
                "--target-minutes",
                str(target_minutes),
                "--avg-wpm",
                f"{avg_wpm:.2f}",
                "--word-budget",
                str(current_word_budget),
                "--word-budget-tolerance",
                str(args.word_budget_tolerance),
                "--agent-tool-mode",
                args.deepseek_agent_tool_mode,
                "--agent-read-max-lines",
                str(args.deepseek_agent_read_max_lines),
                "--agent-max-tool-rounds",
                str(resolved_max_tool_rounds),
                "--agent-loop-exhaustion-policy",
                args.deepseek_agent_loop_exhaustion_policy,
                "--budget-failure-policy",
                args.deepseek_budget_failure_policy,
            ]
            try:
                _run_stage_command(
                    cmd=summarize_cmd,
                    current_env=current_env,
                    use_dashboard=dashboard.enabled,
                    stage_name="Stage 2 (Summarizer)",
                    heartbeat_seconds=args.heartbeat_seconds,
                    model_info=(
                        "Provider: deepseek | Model: auto(reasoner/chat) "
                        f"| Max output tokens: {args.deepseek_max_completion_tokens}"
                    ),
                )
            except subprocess.CalledProcessError as e:
                _print_error(
                    f"[ERROR] Stage 2 crashed with code {e.returncode}",
                    use_rich=use_rich,
                )
                return 1

            if args.skip_stage3:
                break

            if tts_pacing_calibration is not None:
                from audio2script_and_summarizer.tts_pacing_calibration import (
                    estimate_summary_duration_seconds,
                )

                estimated_seconds = estimate_summary_duration_seconds(
                    summary_json_path=summary_output,
                    calibration=tts_pacing_calibration,
                )
                _print_info(
                    f"[INFO] Stage 3 preflight estimate={estimated_seconds:.2f}s for pass {pass_label}.",
                    use_rich=use_rich,
                )

            _print_stage_banner(
                "[START] STAGE 3: Voice Cloning + Interjections",
                use_rich=use_rich,
            )
            _print_checkpoint(
                "Stage 3: running IndexTTS2 synthesis and interjection overlay.",
                use_rich=use_rich,
            )
            stage3_output_override = (
                Path(args.stage3_output).resolve() if args.stage3_output else None
            )
            try:
                (
                    final_wav_path,
                    interjection_log_path,
                    final_mistral_enabled,
                    measured_duration_seconds,
                ) = _run_stage3_from_summary(
                    summary_json_path=Path(summary_output).resolve(),
                    output_wav_path=stage3_output_override,
                    runtime_device=runtime_device,
                    interjection_max_ratio=args.interjection_max_ratio,
                    mistral_model_id=args.mistral_model_id,
                    mistral_max_new_tokens=args.mistral_max_new_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                _print_error(f"[ERROR] Stage 3 failed: {exc}", use_rich=use_rich)
                return 1

            duration_delta_seconds = (
                measured_duration_seconds - target_duration_seconds
            )
            within_tolerance = (
                abs(duration_delta_seconds) <= args.duration_tolerance_seconds
            )
            _print_info(
                (
                    "[INFO] Duration check: "
                    f"target={target_duration_seconds:.2f}s "
                    f"actual={measured_duration_seconds:.2f}s "
                    f"delta={duration_delta_seconds:+.2f}s "
                    f"tolerance={args.duration_tolerance_seconds:.2f}s"
                ),
                use_rich=use_rich,
            )
            if within_tolerance:
                break
            if correction_passes_used >= max_correction_passes:
                _print_warning(
                    (
                        "[WARN] Duration is outside tolerance and max correction "
                        f"passes ({max_correction_passes}) reached."
                    ),
                    use_rich=use_rich,
                )
                break

            corrected_budget = _calculate_corrected_word_budget(
                current_word_budget=current_word_budget,
                target_duration_seconds=target_duration_seconds,
                measured_duration_seconds=measured_duration_seconds,
            )
            if corrected_budget == current_word_budget:
                if measured_duration_seconds > target_duration_seconds:
                    corrected_budget = max(1, current_word_budget - 1)
                else:
                    corrected_budget = current_word_budget + 1
            correction_passes_used += 1
            _print_warning(
                (
                    "[WARN] Applying duration correction: "
                    f"word_budget {current_word_budget} -> {corrected_budget} "
                    f"(correction pass {correction_passes_used}/{max_correction_passes})."
                ),
                use_rich=use_rich,
            )
            current_word_budget = corrected_budget

        dashboard.complete_stage("Stage 2 complete")
        _print_success(f"Summary saved to: {summary_output}", use_rich=use_rich)
        if not args.skip_stage3:
            dashboard.complete_stage("Stage 3 complete")
            _update_summary_report_duration_metrics(
                summary_output_path=summary_output,
                target_duration_seconds=target_duration_seconds,
                measured_duration_seconds=measured_duration_seconds,
                duration_tolerance_seconds=args.duration_tolerance_seconds,
                duration_correction_passes=correction_passes_used,
            )
            if not final_mistral_enabled:
                _print_warning(
                    "[WARN] Stage 3 ran without Mistral interjections because the model was unavailable.",
                    use_rich=use_rich,
                )
            if final_wav_path is not None:
                _print_success(f"Final audio: {final_wav_path}", use_rich=use_rich)
            if interjection_log_path is not None:
                _print_success(
                    f"Interjection log: {interjection_log_path}",
                    use_rich=use_rich,
                )
        else:
            dashboard.complete_stage("Stage 3 skipped (--skip-stage3)")
            _update_summary_report_duration_metrics(
                summary_output_path=summary_output,
                target_duration_seconds=target_duration_seconds,
                measured_duration_seconds=None,
                duration_tolerance_seconds=args.duration_tolerance_seconds,
                duration_correction_passes=0,
            )
            _print_info(
                "[INFO] Stage 3 skipped by --skip-stage3.",
                use_rich=use_rich,
            )
        _print_stage_banner("[DONE] PIPELINE COMPLETE", use_rich=use_rich)
        return 0

    # ==========================================
    # STAGE 1: Diarization
    # ==========================================
    _print_stage_banner("[START] STAGE 1: Audio2Script", use_rich=use_rich)
    _print_checkpoint(
        "Stage 1: preparing diarization subprocess", use_rich=use_rich
    )

    try:
        diarize_cmd = [
            sys.executable,
            "-m",
            "audio2script_and_summarizer.diarize",
            "-a",
            input_path,
            "--device",
            runtime_device,
            "--batch-size",
            "2" if runtime_device.lower().startswith("cuda") else "1",
        ]
        if args.no_stem:
            diarize_cmd.append("--no-stem")
        if args.show_deprecation_warnings:
            diarize_cmd.append("--show-deprecation-warnings")
        if args.plain_ui:
            diarize_cmd.append("--plain-ui")
        if args.no_progress or dashboard.enabled:
            diarize_cmd.append("--no-progress")
        if dashboard.enabled:
            _print_info(
                "[INFO] Stage 1 child progress bars disabled; rendering consolidated progress panel.",
                use_rich=use_rich,
            )

        _run_stage_command(
            cmd=diarize_cmd,
            current_env=current_env,
            use_dashboard=dashboard.enabled,
            stage_name="Stage 1 (Audio2Script)",
            heartbeat_seconds=args.heartbeat_seconds,
            model_info=f"Diarizer: pyannote | Whisper: medium.en | Device: {runtime_device}",
        )
    except subprocess.CalledProcessError as e:
        _print_error(
            f"[ERROR] Stage 1 crashed with code {e.returncode}", use_rich=use_rich
        )
        _print_warning(
            "Tip: If code is -6 or -11, it's a library/driver mismatch.",
            use_rich=use_rich,
        )
        return 1
    dashboard.complete_stage("Stage 1 complete")

    base_name = os.path.splitext(input_path)[0]
    diarization_json = f"{base_name}.json"
    voice_dir = args.voice_dir or f"{base_name}_voices"

    if not os.path.exists(diarization_json):
        _print_error(
            f"[ERROR] Expected output not found: {diarization_json}",
            use_rich=use_rich,
        )
        _print_warning(
            "Did you update diarize.py to export JSON?", use_rich=use_rich
        )
        return 1

    _print_success(
        f"[SUCCESS] Stage 1 Complete. Output: {diarization_json}",
        use_rich=use_rich,
    )

    # ==========================================
    # STAGE 1.5: Audio Splitting
    # ==========================================
    _print_stage_banner("[START] STAGE 1.5: Audio Splitting", use_rich=use_rich)
    _print_checkpoint(
        "Stage 1.5: preparing audio splitting subprocess", use_rich=use_rich
    )

    try:
        splitter_cmd = [
            sys.executable,
            "-m",
            "audio2script_and_summarizer.audio_splitter",
            "--audio",
            input_path,
            "--json",
            diarization_json,
            "--output-dir",
            voice_dir,
        ]
        _run_stage_command(
            cmd=splitter_cmd,
            current_env=current_env,
            use_dashboard=dashboard.enabled,
            stage_name="Stage 1.5 (Audio Splitting)",
            heartbeat_seconds=args.heartbeat_seconds,
            model_info="Module: audio_splitter | Action: extract speaker samples",
        )
    except subprocess.CalledProcessError as e:
        _print_error(
            f"[ERROR] Stage 1.5 crashed with code {e.returncode}",
            use_rich=use_rich,
        )
        return 1
    dashboard.complete_stage("Stage 1.5 complete")

    voice_sample_count = _count_wav_files(voice_dir)
    if voice_sample_count <= 0:
        _print_error(
            f"[ERROR] No speaker samples were generated in: {voice_dir}",
            use_rich=use_rich,
        )
        return 1
    _print_success(
        f"[SUCCESS] Stage 1.5 Complete. Generated {voice_sample_count} speaker sample(s) in {voice_dir}",
        use_rich=use_rich,
    )

    # ==========================================
    # STAGE 1.75: Voice Cloner WPM Calibration
    # ==========================================
    _print_stage_banner(
        "[START] STAGE 1.75: Voice Cloner WPM Calibration", use_rich=use_rich
    )
    _print_checkpoint(
        "Stage 1.75: calibrating voice WPM (this can take a while)",
        use_rich=use_rich,
    )
    _print_checkpoint(
        f"Stage 1.75: WPM source selected: {normalized_wpm_source}",
        use_rich=use_rich,
    )

    stage_175_name = "Stage 1.75 (WPM Calibration)"
    stage_175_module = (
        "audio2script_and_summarizer.tts_pacing_calibration"
        if normalized_wpm_source == "tts_preflight"
        else "audio2script_and_summarizer.transcript_wpm"
    )
    stage_175_command = (
        "calibrate_tts_pacing_profiles(...)"
        if normalized_wpm_source == "tts_preflight"
        else "compute_wpm_from_transcript(...)"
    )
    stage_175_model_info = (
        "Emotion-aware IndexTTS2 checkpoint preflight calibration"
        if normalized_wpm_source == "tts_preflight"
        else "Diarized transcript timestamp-based WPM derivation"
    )
    if dashboard.enabled:
        dashboard.set_status(
            stage_name=stage_175_name,
            substep="Running in-process calibration",
            module_name=stage_175_module,
            command_display=stage_175_command,
            model_info=stage_175_model_info,
            pid=None,
            reset_elapsed=True,
        )
        detail_label = (
            "IndexTTS2 preflight pacing calibration"
            if normalized_wpm_source == "tts_preflight"
            else "Transcript WPM derivation"
        )
        dashboard.start_detail_progress(detail_label)
        dashboard.update_detail_progress(1)

    per_speaker_wpm = {}
    tts_pacing_calibration = None
    try:
        if normalized_wpm_source == "tts_preflight":
            from audio2script_and_summarizer.tts_pacing_calibration import (
                CalibrationEvent,
            )

            def _on_calibration_progress(event: CalibrationEvent) -> None:
                """Reflect TTS preflight calibration progress in the dashboard."""
                if not dashboard.enabled:
                    return
                if event.event_type == "model_init_started":
                    dashboard.set_status(
                        stage_name=stage_175_name,
                        substep="Initializing IndexTTS2 artifacts",
                        module_name=stage_175_module,
                        command_display=stage_175_command,
                        model_info=stage_175_model_info,
                        pid=None,
                    )
                    dashboard.update_detail_progress(5)
                    return
                if event.event_type == "model_init_completed":
                    speaker_total = max(1, event.speaker_count or 1)
                    dashboard.set_status(
                        stage_name=stage_175_name,
                        substep=f"Model ready; calibrating {speaker_total} speaker sample(s)",
                        module_name=stage_175_module,
                        command_display=stage_175_command,
                        model_info=stage_175_model_info,
                        pid=None,
                    )
                    dashboard.update_detail_progress(12)
                    return
                if event.event_type == "speaker_started":
                    speaker_total = max(1, event.speaker_count or 1)
                    speaker_index = max(1, event.speaker_index or 1)
                    progress_floor = 12
                    progress_span = 84
                    progress = progress_floor + int(
                        ((speaker_index - 1) / speaker_total) * progress_span
                    )
                    speaker_display = (
                        event.speaker_name or f"speaker_{speaker_index}"
                    )
                    dashboard.set_status(
                        stage_name=stage_175_name,
                        substep=f"Calibrating {speaker_display} ({speaker_index}/{speaker_total})",
                        module_name=stage_175_module,
                        command_display=stage_175_command,
                        model_info=stage_175_model_info,
                        pid=None,
                    )
                    dashboard.update_detail_progress(progress)
                    return
                if event.event_type == "speaker_completed":
                    speaker_total = max(1, event.speaker_count or 1)
                    speaker_index = max(1, event.speaker_index or 1)
                    progress_floor = 12
                    progress_span = 84
                    progress = progress_floor + int(
                        (speaker_index / speaker_total) * progress_span
                    )
                    speaker_display = (
                        event.speaker_name or f"speaker_{speaker_index}"
                    )
                    speaker_wpm = event.speaker_wpm or 0.0
                    dashboard.set_status(
                        stage_name=stage_175_name,
                        substep=f"Calibrated {speaker_display} ({speaker_wpm:.2f} WPM)",
                        module_name=stage_175_module,
                        command_display=stage_175_command,
                        model_info=stage_175_model_info,
                        pid=None,
                    )
                    dashboard.update_detail_progress(progress)
                    return
                if event.event_type == "calibration_completed":
                    avg_wpm_local = event.average_wpm or 0.0
                    dashboard.set_status(
                        stage_name=stage_175_name,
                        substep=f"Calibration complete (avg {avg_wpm_local:.2f} WPM)",
                        module_name=stage_175_module,
                        command_display=stage_175_command,
                        model_info=stage_175_model_info,
                        pid=None,
                    )
                    dashboard.update_detail_progress(100)

            capture_stage175_runtime_output = (
                _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled
            )
            (
                avg_wpm,
                per_speaker_wpm,
                tts_pacing,
                cache_hit,
                cache_path,
            ) = _compute_tts_preflight_wpm(
                args=args,
                voice_dir=voice_dir,
                runtime_device=runtime_device,
                transcript_json_path=diarization_json,
                capture_stage175_runtime_output=capture_stage175_runtime_output,
                progress_cb=(
                    _on_calibration_progress if dashboard.enabled else None
                ),
                _capture_stage175_output_lines=_capture_stage175_output_lines,
                logger=logger,
            )
            tts_pacing_calibration = tts_pacing
            if cache_hit and cache_path is not None:
                logger.info("Stage 1.75 reused calibration cache at %s", cache_path)
                if dashboard.enabled:
                    dashboard.set_status(
                        stage_name=stage_175_name,
                        substep="Loaded cached IndexTTS2 pacing calibration",
                        module_name=stage_175_module,
                        command_display=stage_175_command,
                        model_info=stage_175_model_info,
                        pid=None,
                    )
                    dashboard.update_detail_progress(100)
            elif cache_path is not None:
                logger.info("Stage 1.75 updated calibration cache at %s", cache_path)
            logger.info(
                "Stage 1.75 weighted WPM derived from TTS preflight + transcript distribution: avg_wpm=%.2f",
                avg_wpm,
            )
        else:
            from audio2script_and_summarizer.transcript_wpm import (
                compute_wpm_from_transcript,
            )

            if dashboard.enabled:
                dashboard.set_status(
                    stage_name=stage_175_name,
                    substep="Parsing diarized transcript segments",
                    module_name=stage_175_module,
                    command_display=stage_175_command,
                    model_info=stage_175_model_info,
                    pid=None,
                )
                dashboard.update_detail_progress(40)
            avg_wpm, per_speaker_wpm = compute_wpm_from_transcript(diarization_json)
            if dashboard.enabled:
                dashboard.set_status(
                    stage_name=stage_175_name,
                    substep=f"Derived transcript WPM (avg {avg_wpm:.2f})",
                    module_name=stage_175_module,
                    command_display=stage_175_command,
                    model_info=stage_175_model_info,
                    pid=None,
                )
                dashboard.update_detail_progress(100)
    except Exception as exc:  # noqa: BLE001
        _print_error(f"[ERROR] WPM calibration failed: {exc}", use_rich=use_rich)
        return 1

    dashboard.complete_stage("Stage 1.75 complete")

    _print_info(
        f"[INFO] Stage 1.75 source={normalized_wpm_source}; per-speaker WPM: "
        f"{_format_speaker_wpm_summary(per_speaker_wpm)}",
        use_rich=use_rich,
    )

    if target_minutes is None:
        _print_error(
            "[ERROR] target minutes is required for full pipeline mode.",
            use_rich=use_rich,
        )
        return 1
    word_budget = max(1, int(round(avg_wpm * target_minutes)))
    _print_success(
        f"[SUCCESS] WPM source={normalized_wpm_source}; calibrated WPM={avg_wpm:.2f}; "
        f"target_minutes={target_minutes:.2f}; word_budget={word_budget}",
        use_rich=use_rich,
    )

    # ==========================================
    # STAGE 2: Summarizer
    # ==========================================
    _print_stage_banner("[START] STAGE 2: Summarizer", use_rich=use_rich)
    _print_checkpoint("Stage 2: preparing summarizer subprocess", use_rich=use_rich)

    summary_output = f"{base_name}_summary.json"
    target_duration_seconds = target_minutes * 60.0
    current_word_budget = word_budget
    correction_passes_used = 0
    max_correction_passes = args.max_duration_correction_passes
    measured_duration_seconds = None
    final_wav_path = None
    interjection_log_path = None
    final_mistral_enabled = True

    while True:
        pass_label = correction_passes_used + 1
        _print_checkpoint(
            (
                f"Stage 2 pass {pass_label}: running summarizer with "
                f"word_budget={current_word_budget}"
            ),
            use_rich=use_rich,
        )
        try:
            if llm_provider == "openai":
                api_key = (
                    args.openai_key
                    or os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("LLM_API_KEY")
                    or os.environ.get("GEMINI_API_KEY")
                )
                if not api_key:
                    _print_error(
                        "[ERROR] No OpenAI API key found. Use --openai-key/--api-key or set OPENAI_API_KEY.",
                        use_rich=use_rich,
                    )
                    return 1
                summarize_cmd = [
                    sys.executable,
                    "-m",
                    "audio2script_and_summarizer.summarizer",
                    "--transcript",
                    diarization_json,
                    "--voice-dir",
                    voice_dir,
                    "--output",
                    summary_output,
                    "--api-key",
                    api_key,
                    "--target-minutes",
                    str(target_minutes),
                    "--avg-wpm",
                    f"{avg_wpm:.2f}",
                    "--word-budget",
                    str(current_word_budget),
                    "--word-budget-tolerance",
                    str(args.word_budget_tolerance),
                ]
                summary_model_info = "Provider: openai | Model: gpt-4o-2024-08-06"
            else:
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                    _print_error(
                        "[ERROR] No DeepSeek API key found. Set DEEPSEEK_API_KEY.",
                        use_rich=use_rich,
                    )
                    return 1
                resolved_max_tool_rounds, rounds_source = (
                    _resolve_deepseek_agent_max_tool_rounds(
                        configured_max_tool_rounds=(
                            args.deepseek_agent_max_tool_rounds
                        ),
                        current_word_budget=current_word_budget,
                        target_minutes=target_minutes,
                    )
                )
                _print_info(
                    (
                        f"[INFO] Stage 2 pass {pass_label}: "
                        f"DeepSeek max_tool_rounds={resolved_max_tool_rounds} "
                        f"(source={rounds_source}; "
                        f"word_budget={current_word_budget}; "
                        f"target_minutes={target_minutes:.2f})."
                    ),
                    use_rich=use_rich,
                )
                summarize_cmd = [
                    sys.executable,
                    "-m",
                    "audio2script_and_summarizer.summarizer_deepseek",
                    "--transcript",
                    diarization_json,
                    "--voice-dir",
                    voice_dir,
                    "--output",
                    summary_output,
                    "--api-key",
                    api_key,
                    "--max-completion-tokens",
                    str(args.deepseek_max_completion_tokens),
                    "--target-minutes",
                    str(target_minutes),
                    "--avg-wpm",
                    f"{avg_wpm:.2f}",
                    "--word-budget",
                    str(current_word_budget),
                    "--word-budget-tolerance",
                    str(args.word_budget_tolerance),
                    "--agent-tool-mode",
                    args.deepseek_agent_tool_mode,
                    "--agent-read-max-lines",
                    str(args.deepseek_agent_read_max_lines),
                    "--agent-max-tool-rounds",
                    str(resolved_max_tool_rounds),
                    "--agent-loop-exhaustion-policy",
                    args.deepseek_agent_loop_exhaustion_policy,
                    "--budget-failure-policy",
                    args.deepseek_budget_failure_policy,
                ]
                summary_model_info = (
                    "Provider: deepseek | Model: auto(reasoner/chat) "
                    f"| Max output tokens: {args.deepseek_max_completion_tokens}"
                )
            _run_stage_command(
                cmd=summarize_cmd,
                current_env=current_env,
                use_dashboard=dashboard.enabled,
                stage_name="Stage 2 (Summarizer)",
                heartbeat_seconds=args.heartbeat_seconds,
                model_info=summary_model_info,
            )
        except subprocess.CalledProcessError as e:
            _print_error(
                f"[ERROR] Stage 2 crashed with code {e.returncode}",
                use_rich=use_rich,
            )
            return 1

        if args.skip_stage3:
            break

        if tts_pacing_calibration is not None:
            from audio2script_and_summarizer.tts_pacing_calibration import (
                estimate_summary_duration_seconds,
            )

            estimated_seconds = estimate_summary_duration_seconds(
                summary_json_path=summary_output,
                calibration=tts_pacing_calibration,
            )
            _print_info(
                f"[INFO] Stage 3 preflight estimate={estimated_seconds:.2f}s for pass {pass_label}.",
                use_rich=use_rich,
            )

        _print_stage_banner(
            "[START] STAGE 3: Voice Cloning + Interjections",
            use_rich=use_rich,
        )
        _print_checkpoint(
            "Stage 3: running IndexTTS2 synthesis and interjection overlay.",
            use_rich=use_rich,
        )
        stage3_output_override = (
            Path(args.stage3_output).resolve() if args.stage3_output else None
        )
        try:
            (
                final_wav_path,
                interjection_log_path,
                final_mistral_enabled,
                measured_duration_seconds,
            ) = _run_stage3_from_summary(
                summary_json_path=Path(summary_output).resolve(),
                output_wav_path=stage3_output_override,
                runtime_device=runtime_device,
                interjection_max_ratio=args.interjection_max_ratio,
                mistral_model_id=args.mistral_model_id,
                mistral_max_new_tokens=args.mistral_max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            _print_error(f"[ERROR] Stage 3 failed: {exc}", use_rich=use_rich)
            return 1

        duration_delta_seconds = measured_duration_seconds - target_duration_seconds
        within_tolerance = (
            abs(duration_delta_seconds) <= args.duration_tolerance_seconds
        )
        _print_info(
            (
                "[INFO] Duration check: "
                f"target={target_duration_seconds:.2f}s "
                f"actual={measured_duration_seconds:.2f}s "
                f"delta={duration_delta_seconds:+.2f}s "
                f"tolerance={args.duration_tolerance_seconds:.2f}s"
            ),
            use_rich=use_rich,
        )
        if within_tolerance:
            _print_success(
                f"[SUCCESS] Duration target met on pass {pass_label}.",
                use_rich=use_rich,
            )
            break
        if correction_passes_used >= max_correction_passes:
            _print_warning(
                (
                    "[WARN] Duration is outside tolerance and max correction passes "
                    f"({max_correction_passes}) reached."
                ),
                use_rich=use_rich,
            )
            break

        corrected_budget = _calculate_corrected_word_budget(
            current_word_budget=current_word_budget,
            target_duration_seconds=target_duration_seconds,
            measured_duration_seconds=measured_duration_seconds,
        )
        if corrected_budget == current_word_budget:
            if measured_duration_seconds > target_duration_seconds:
                corrected_budget = max(1, current_word_budget - 1)
            else:
                corrected_budget = current_word_budget + 1

        correction_passes_used += 1
        _print_warning(
            (
                "[WARN] Applying duration correction: "
                f"word_budget {current_word_budget} -> {corrected_budget} "
                f"(correction pass {correction_passes_used}/{max_correction_passes})."
            ),
            use_rich=use_rich,
        )
        current_word_budget = corrected_budget

    dashboard.complete_stage("Stage 2 complete")
    _print_success(f"Summary saved to: {summary_output}", use_rich=use_rich)

    if not args.skip_stage3:
        dashboard.complete_stage("Stage 3 complete")
        _update_summary_report_duration_metrics(
            summary_output_path=summary_output,
            target_duration_seconds=target_duration_seconds,
            measured_duration_seconds=measured_duration_seconds,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
            duration_correction_passes=correction_passes_used,
        )
        if not final_mistral_enabled:
            _print_warning(
                "[WARN] Stage 3 ran without Mistral interjections because the model was unavailable.",
                use_rich=use_rich,
            )
        if final_wav_path is not None:
            _print_success(f"Final audio: {final_wav_path}", use_rich=use_rich)
        if interjection_log_path is not None:
            _print_success(
                f"Interjection log: {interjection_log_path}",
                use_rich=use_rich,
            )
    else:
        dashboard.complete_stage("Stage 3 skipped (--skip-stage3)")
        _update_summary_report_duration_metrics(
            summary_output_path=summary_output,
            target_duration_seconds=target_duration_seconds,
            measured_duration_seconds=None,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
            duration_correction_passes=0,
        )
        _print_info("[INFO] Stage 3 skipped by --skip-stage3.", use_rich=use_rich)
    _print_stage_banner("[DONE] PIPELINE COMPLETE", use_rich=use_rich)
    return 0

