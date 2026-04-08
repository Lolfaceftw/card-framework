"""Textual operator frontend for setup-and-run and benchmark launches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    OptionList,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from card_framework.cli.frontend_models import BenchmarkKind, FrontendResult
from card_framework.cli.frontend_services import (
    BENCHMARK_KIND_OPTIONS,
    DEFAULT_BENCHMARK_KIND,
    DEFAULT_DURATION_SWEEP_PRESET_LABELS,
    PIPELINE_MODE_OPTIONS,
    STAGE_ROUTING_OPTIONS,
    SUPPORTED_AUDIO_PATH_SUFFIXES,
    compute_path_suggestions,
    FrontendDefaults,
    FrontendValidationError,
    build_benchmark_request,
    build_pipeline_selection,
    describe_benchmark_request,
    describe_pipeline_selection,
    load_frontend_defaults,
    save_config_text,
    validate_config_text,
)


class SetupAndRunFrontendApp(App[FrontendResult]):
    """Textual dashboard for pipeline, benchmark, and config workflows."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #intro {
        padding: 0 1;
        color: $text-muted;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 0 1;
    }

    VerticalScroll {
        padding-right: 1;
    }

    .field-label {
        margin: 1 0 0 0;
        color: $text-muted;
    }

    Input, Select, TextArea {
        width: 1fr;
    }

    TextArea {
        min-height: 6;
    }

    OptionList.path-suggestions {
        height: auto;
        max-height: 8;
        margin: 0 0 1 0;
        border: round $panel;
    }

    #config_editor {
        height: 1fr;
        min-height: 20;
    }

    .preview {
        border: round $primary;
        padding: 1;
        margin: 1 0;
        min-height: 8;
    }

    .button-row {
        height: auto;
        margin: 1 0;
    }

    .form-group {
        height: auto;
    }

    Button {
        margin-right: 1;
    }
    """

    BINDINGS = [
        ("ctrl+s", "save_config", "Save config"),
        ("q", "quit_frontend", "Quit"),
    ]

    def __init__(self, *, defaults: FrontendDefaults, uv_executable: str) -> None:
        super().__init__()
        self._defaults = defaults
        self._uv_executable = uv_executable
        self._config_dirty = False
        self._workspace_root = defaults.workspace_root.resolve()
        self._path_suggestion_values: dict[str, tuple[str, ...]] = {}

    @staticmethod
    def _as_select_options(
        options: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    ) -> list[tuple[str, str]]:
        return [(label, value) for value, label in options]

    def compose(self) -> ComposeResult:
        stage_profile_options = [
            ("Use the shared LLM", "default"),
            *self._as_select_options(list(self._defaults.llm_profiles)),
        ]
        yield Header()
        yield Static(
            "Pick what you want to do, confirm the exact inputs, then stay in a "
            "Textual operator view while the selected workflow runs.",
            id="intro",
        )
        with TabbedContent(initial="pipeline"):
            with TabPane("Pipeline", id="pipeline"):
                with VerticalScroll():
                    with Vertical(id="pipeline_audio_group", classes="form-group"):
                        yield Static(
                            "Source audio file",
                            classes="field-label",
                        )
                        yield Input(
                            value=self._defaults.discovered_audio_path,
                            id="pipeline_primary_audio_path",
                            placeholder="audio.wav",
                        )
                        yield OptionList(
                            id="pipeline_primary_audio_suggestions",
                            classes="path-suggestions",
                            markup=False,
                        )
                    yield Static("Run mode", classes="field-label")
                    yield Select(
                        self._as_select_options(PIPELINE_MODE_OPTIONS),
                        value="stage1_single",
                        allow_blank=False,
                        id="pipeline_mode",
                    )
                    with Vertical(
                        id="pipeline_additional_audio_group",
                        classes="form-group",
                    ):
                        yield Static(
                            "Additional audio files for batch runs (one path per line)",
                            classes="field-label",
                        )
                        yield TextArea(
                            id="pipeline_additional_audio_paths",
                            placeholder="episode-b.mp3\nepisode-c.wav",
                        )
                    with Vertical(id="pipeline_duration_group", classes="form-group"):
                        yield Static(
                            "Duration presets (one per line)", classes="field-label"
                        )
                        yield TextArea(
                            "\n".join(DEFAULT_DURATION_SWEEP_PRESET_LABELS),
                            id="pipeline_duration_presets",
                        )
                    with Vertical(id="pipeline_stage2_group", classes="form-group"):
                        yield Static("Transcript JSON path", classes="field-label")
                        yield Input(
                            value=self._defaults.discovered_transcript_path,
                            id="pipeline_transcript_path",
                        )
                        yield OptionList(
                            id="pipeline_transcript_suggestions",
                            classes="path-suggestions",
                            markup=False,
                        )
                        yield Static(
                            "Optional source audio path for stage-2 fallback",
                            classes="field-label",
                        )
                        yield Input(
                            id="pipeline_stage2_audio_path",
                            placeholder="audio.wav",
                        )
                        yield OptionList(
                            id="pipeline_stage2_audio_suggestions",
                            classes="path-suggestions",
                            markup=False,
                        )
                    with Vertical(id="pipeline_stage3_group", classes="form-group"):
                        yield Static("Summary XML path", classes="field-label")
                        yield Input(
                            value=self._defaults.discovered_summary_path,
                            id="pipeline_summary_path",
                        )
                        yield OptionList(
                            id="pipeline_summary_suggestions",
                            classes="path-suggestions",
                            markup=False,
                        )
                    with Vertical(id="pipeline_stage4_group", classes="form-group"):
                        yield Static("Voice-clone manifest path", classes="field-label")
                        yield Input(id="pipeline_voice_clone_manifest_path")
                        yield OptionList(
                            id="pipeline_voice_clone_manifest_suggestions",
                            classes="path-suggestions",
                            markup=False,
                        )
                    with Vertical(id="pipeline_resume_group", classes="form-group"):
                        yield Static("Session UUID", classes="field-label")
                        yield Input(id="pipeline_resume_session")
                    yield Static("Output root (optional)", classes="field-label")
                    yield Input(
                        id="pipeline_output_root",
                        placeholder="Leave blank to use the workspace default",
                    )
                    yield Static("Shared LLM profile", classes="field-label")
                    yield Select(
                        self._as_select_options(list(self._defaults.llm_profiles)),
                        value=self._defaults.pipeline_defaults.llm,
                        allow_blank=False,
                        id="pipeline_shared_llm",
                    )
                    yield Static("Stage routing", classes="field-label")
                    yield Select(
                        self._as_select_options(STAGE_ROUTING_OPTIONS),
                        value="config_default",
                        allow_blank=False,
                        id="pipeline_stage_strategy",
                    )
                    with Vertical(
                        id="pipeline_custom_stage_group", classes="form-group"
                    ):
                        yield Static("Summarizer LLM profile", classes="field-label")
                        yield Select(
                            stage_profile_options,
                            value=self._defaults.pipeline_defaults.stage_llm[
                                "summarizer"
                            ],
                            allow_blank=False,
                            id="pipeline_stage_llm_summarizer",
                        )
                        yield Static("Critic LLM profile", classes="field-label")
                        yield Select(
                            stage_profile_options,
                            value=self._defaults.pipeline_defaults.stage_llm["critic"],
                            allow_blank=False,
                            id="pipeline_stage_llm_critic",
                        )
                        yield Static("Interjector LLM profile", classes="field-label")
                        yield Select(
                            stage_profile_options,
                            value=self._defaults.pipeline_defaults.stage_llm[
                                "interjector"
                            ],
                            allow_blank=False,
                            id="pipeline_stage_llm_interjector",
                        )
                    yield Static("Embedding profile", classes="field-label")
                    yield Select(
                        self._as_select_options(
                            list(self._defaults.embedding_profiles)
                        ),
                        value=self._defaults.pipeline_defaults.embedding,
                        allow_blank=False,
                        id="pipeline_embedding",
                    )
                    with Horizontal(classes="button-row"):
                        yield Checkbox(
                            "Enable voice clone",
                            value=self._defaults.pipeline_defaults.voice_clone_enabled,
                            id="pipeline_voice_clone_enabled",
                        )
                        yield Checkbox(
                            "Enable interjector",
                            value=self._defaults.pipeline_defaults.interjector_enabled,
                            id="pipeline_interjector_enabled",
                        )
                    yield Static(
                        "Extra Hydra overrides (one per line)", classes="field-label"
                    )
                    yield TextArea(
                        id="pipeline_extra_overrides",
                        placeholder="selection.llm=anthropic_claude_opus",
                    )
                    yield Static(id="pipeline_preview", classes="preview")
                    with Horizontal(classes="button-row"):
                        yield Button(
                            "Launch pipeline", id="launch_pipeline", variant="primary"
                        )
                        yield Button("Cancel", id="cancel_pipeline")
            with TabPane("Benchmarks", id="benchmarks"):
                with VerticalScroll():
                    yield Static(
                        "Benchmark or experiment target", classes="field-label"
                    )
                    yield Select(
                        self._as_select_options(BENCHMARK_KIND_OPTIONS),
                        value=DEFAULT_BENCHMARK_KIND,
                        allow_blank=False,
                        id="benchmark_kind",
                    )
                    yield Static(
                        "Output directory/root (optional)", classes="field-label"
                    )
                    yield Input(
                        id="benchmark_output_dir",
                        placeholder="Leave blank to use the CLI default",
                    )
                    with Vertical(id="benchmark_group_summary_matrix"):
                        yield Static("vLLM host", classes="field-label")
                        yield Input(id="matrix_vllm_host", placeholder="127.0.0.1")
                        yield Static("Matrix input mode", classes="field-label")
                        yield Select(
                            (
                                ("Audio path", "audio"),
                                ("Transcript path", "transcript"),
                            ),
                            value="audio",
                            allow_blank=False,
                            id="matrix_input_mode",
                        )
                        yield Static("Matrix input path", classes="field-label")
                        yield Input(
                            id="matrix_input_path",
                            placeholder="audio.wav",
                        )
                        yield OptionList(
                            id="matrix_input_suggestions",
                            classes="path-suggestions",
                            markup=False,
                        )
                        yield Static("Optional DeepSeek API key", classes="field-label")
                        yield Input(id="matrix_deepseek_api_key", password=True)
                        yield Static("Optional DeepSeek model", classes="field-label")
                        yield Input(value="deepseek-chat", id="matrix_deepseek_model")
                        yield Static(
                            "Optional DeepSeek base URL", classes="field-label"
                        )
                        yield Input(
                            value="https://api.deepseek.com",
                            id="matrix_deepseek_base_url",
                        )
                        yield Checkbox(
                            "Fail fast on first model-pair failure",
                            id="matrix_fail_fast",
                        )
                    with Vertical(id="benchmark_group_summarizer_benchmark"):
                        yield Static("Benchmark manifest path", classes="field-label")
                        yield Input(
                            value=self._defaults.summary_benchmark_manifest,
                            id="summarizer_manifest",
                        )
                        yield Static("Preset", classes="field-label")
                        yield Input(value="hourly", id="summarizer_preset")
                        yield Static(
                            "Providers (comma-separated, optional)",
                            classes="field-label",
                        )
                        yield Input(id="summarizer_providers")
                        yield Static("Judge provider (optional)", classes="field-label")
                        yield Input(id="summarizer_judge_provider")
                        yield Static("Max samples (optional)", classes="field-label")
                        yield Input(id="summarizer_max_samples")
                    with Vertical(id="benchmark_group_qa_benchmark"):
                        yield Static("Summary XML path", classes="field-label")
                        yield Input(
                            value=self._defaults.discovered_summary_path,
                            id="qa_summary_xml",
                        )
                        yield Static("Source transcript path", classes="field-label")
                        yield Input(
                            value=self._defaults.discovered_transcript_path,
                            id="qa_source_transcript",
                        )
                        yield Static("QA config path", classes="field-label")
                        yield Input(
                            value=self._defaults.qa_config_path, id="qa_config_path"
                        )
                        yield Static("Default provider", classes="field-label")
                        yield Input(value="vllm_default", id="qa_provider")
                        yield Static(
                            "Creator provider override (optional)",
                            classes="field-label",
                        )
                        yield Input(id="qa_creator_provider")
                        yield Static(
                            "Evaluator provider override (optional)",
                            classes="field-label",
                        )
                        yield Input(id="qa_evaluator_provider")
                        yield Static("Output format", classes="field-label")
                        yield Select(
                            (
                                ("auto", "auto"),
                                ("human", "human"),
                                ("json", "json"),
                                ("both", "both"),
                            ),
                            value="auto",
                            allow_blank=False,
                            id="qa_output_format",
                        )
                        yield Static("Color", classes="field-label")
                        yield Select(
                            (
                                ("auto", "auto"),
                                ("always", "always"),
                                ("never", "never"),
                            ),
                            value="auto",
                            allow_blank=False,
                            id="qa_color",
                        )
                    with Vertical(id="benchmark_group_qmsum_benchmark"):
                        yield Static(
                            "Prepared QMSum dataset manifest", classes="field-label"
                        )
                        yield Input(
                            value=self._defaults.qmsum_dataset_manifest,
                            id="qmsum_dataset_manifest",
                        )
                        yield Static("Summarizer provider", classes="field-label")
                        yield Input(
                            value="vllm_default", id="qmsum_summarizer_provider"
                        )
                        yield Static(
                            "Critic provider (optional)", classes="field-label"
                        )
                        yield Input(id="qmsum_critic_provider")
                        yield Static("Judge provider (optional)", classes="field-label")
                        yield Input(id="qmsum_judge_provider")
                        yield Static("Embedding mode", classes="field-label")
                        yield Select(
                            (("disabled", "disabled"), ("enabled", "enabled")),
                            value="disabled",
                            allow_blank=False,
                            id="qmsum_embedding_mode",
                        )
                        yield Static("Max samples (optional)", classes="field-label")
                        yield Input(id="qmsum_max_samples")
                        yield Static("Judge repeats (optional)", classes="field-label")
                        yield Input(id="qmsum_judge_repeats")
                        yield Static("Max iterations (optional)", classes="field-label")
                        yield Input(id="qmsum_max_iterations")
                        yield Checkbox(
                            "Prepare dataset if missing", id="qmsum_prepare_if_missing"
                        )
                        yield Checkbox(
                            "Force dataset download", id="qmsum_force_download_dataset"
                        )
                    with Vertical(id="benchmark_group_sporc_benchmark"):
                        yield Static(
                            "Prepared SPoRC dataset manifest", classes="field-label"
                        )
                        yield Input(
                            value=self._defaults.sporc_dataset_manifest,
                            id="sporc_dataset_manifest",
                        )
                        yield Static("Dataset variant", classes="field-label")
                        yield Select(
                            (("sample", "sample"), ("full", "full")),
                            value="sample",
                            allow_blank=False,
                            id="sporc_dataset_variant",
                        )
                        yield Static("Summarizer provider", classes="field-label")
                        yield Input(
                            value="vllm_default", id="sporc_summarizer_provider"
                        )
                        yield Static(
                            "Critic provider (optional)", classes="field-label"
                        )
                        yield Input(id="sporc_critic_provider")
                        yield Static("Judge provider (optional)", classes="field-label")
                        yield Input(id="sporc_judge_provider")
                        yield Static("Embedding mode", classes="field-label")
                        yield Select(
                            (("disabled", "disabled"), ("enabled", "enabled")),
                            value="disabled",
                            allow_blank=False,
                            id="sporc_embedding_mode",
                        )
                        yield Static(
                            "Duration preset (optional)", classes="field-label"
                        )
                        yield Input(
                            id="sporc_duration_preset",
                            placeholder="30s, 1m, 5m, 10m, 15m",
                        )
                        yield Static("Max samples (optional)", classes="field-label")
                        yield Input(id="sporc_max_samples")
                        yield Static("Max iterations (optional)", classes="field-label")
                        yield Input(id="sporc_max_iterations")
                        yield Checkbox(
                            "Transcript only", value=True, id="sporc_transcript_only"
                        )
                    with Vertical(id="benchmark_group_sporc_duration_sweep"):
                        yield Static("Sweep root", classes="field-label")
                        yield Input(
                            value=self._defaults.sporc_sweep_root, id="sporc_sweep_root"
                        )
                        yield Static("Sweep mode", classes="field-label")
                        yield Select(
                            (
                                ("menu", "menu"),
                                ("single", "single"),
                                ("batch-all", "batch-all"),
                            ),
                            value="menu",
                            allow_blank=False,
                            id="sporc_sweep_mode",
                        )
                        yield Static(
                            "Duration presets (comma-separated)", classes="field-label"
                        )
                        yield Input(
                            value="30s,1m,5m", id="sporc_sweep_duration_presets"
                        )
                        yield Static(
                            "Examples / max samples (optional)", classes="field-label"
                        )
                        yield Input(id="sporc_sweep_examples")
                        yield Static("Parallelism (optional)", classes="field-label")
                        yield Input(id="sporc_sweep_parallelism")
                        yield Static(
                            "Per-preset concurrency (optional)", classes="field-label"
                        )
                        yield Input(id="sporc_sweep_concurrency")
                        yield Checkbox(
                            "Enable voice clone", id="sporc_sweep_enable_voice_clone"
                        )
                    with Vertical(id="benchmark_group_diarization_benchmark"):
                        yield Static("Diarization manifest", classes="field-label")
                        yield Input(
                            value=self._defaults.diarization_manifest,
                            id="diarization_manifest",
                        )
                        yield Static(
                            "Providers (comma-separated, optional)",
                            classes="field-label",
                        )
                        yield Input(id="diarization_providers")
                        yield Static("Device", classes="field-label")
                        yield Select(
                            (("auto", "auto"), ("cpu", "cpu"), ("cuda", "cuda")),
                            value="auto",
                            allow_blank=False,
                            id="diarization_device",
                        )
                        yield Static("Max samples (optional)", classes="field-label")
                        yield Input(id="diarization_max_samples")
                        yield Static("Collar seconds (optional)", classes="field-label")
                        yield Input(id="diarization_collar")
                        yield Checkbox("Skip overlap", id="diarization_skip_overlap")
                    yield Static(
                        "Extra CLI arguments (one per line)", classes="field-label"
                    )
                    yield TextArea(
                        id="benchmark_extra_args", placeholder="--sample-seed=42"
                    )
                    yield Static(id="benchmark_preview", classes="preview")
                    with Horizontal(classes="button-row"):
                        yield Button(
                            "Launch benchmark", id="launch_benchmark", variant="primary"
                        )
                        yield Button("Cancel", id="cancel_benchmark")
            with TabPane("Config", id="config"):
                with VerticalScroll():
                    yield Static(
                        f"Active config: {self._defaults.config_path}",
                        classes="field-label",
                    )
                    yield TextArea(
                        self._defaults.config_text, language="yaml", id="config_editor"
                    )
                    yield Static(id="config_status", classes="preview")
                    with Horizontal(classes="button-row"):
                        yield Button(
                            "Save config", id="save_config_button", variant="success"
                        )
                        yield Button("Reload from disk", id="reload_config_button")
                        yield Button("Close frontend", id="close_frontend")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_pipeline_visibility()
        self._refresh_benchmark_visibility()
        self._refresh_all_path_suggestions()
        self._refresh_pipeline_preview()
        self._refresh_benchmark_preview()
        self._refresh_config_status()

    def _bool(self, widget_id: str) -> bool:
        return bool(self.query_one(f"#{widget_id}", Checkbox).value)

    def _text(self, widget_id: str) -> str:
        widget = self.query_one(f"#{widget_id}")
        if isinstance(widget, Input):
            return widget.value
        if isinstance(widget, TextArea):
            return widget.text
        raise TypeError(f"Unsupported text widget: {widget_id}")

    def _select_value(self, widget_id: str) -> str:
        return str(self.query_one(f"#{widget_id}", Select).value)

    def _pipeline_values(self) -> dict[str, Any]:
        return {
            "mode": self._select_value("pipeline_mode"),
            "output_root": self._text("pipeline_output_root"),
            "shared_llm": self._select_value("pipeline_shared_llm"),
            "stage_strategy": self._select_value("pipeline_stage_strategy"),
            "stage_llm_summarizer": self._select_value("pipeline_stage_llm_summarizer"),
            "stage_llm_critic": self._select_value("pipeline_stage_llm_critic"),
            "stage_llm_interjector": self._select_value(
                "pipeline_stage_llm_interjector"
            ),
            "embedding": self._select_value("pipeline_embedding"),
            "voice_clone_enabled": self._bool("pipeline_voice_clone_enabled"),
            "interjector_enabled": self._bool("pipeline_interjector_enabled"),
            "primary_audio_path": self._text("pipeline_primary_audio_path"),
            "additional_audio_paths": self._text("pipeline_additional_audio_paths"),
            "duration_presets": self._text("pipeline_duration_presets"),
            "transcript_path": self._text("pipeline_transcript_path"),
            "stage2_audio_path": self._text("pipeline_stage2_audio_path"),
            "summary_path": self._text("pipeline_summary_path"),
            "voice_clone_manifest_path": self._text(
                "pipeline_voice_clone_manifest_path"
            ),
            "resume_session": self._text("pipeline_resume_session"),
            "extra_overrides": self._text("pipeline_extra_overrides"),
        }

    def _benchmark_values(self) -> dict[str, Any]:
        return {
            "benchmark_output_dir": self._text("benchmark_output_dir"),
            "benchmark_extra_args": self._text("benchmark_extra_args"),
            "matrix_vllm_host": self._text("matrix_vllm_host"),
            "matrix_input_mode": self._select_value("matrix_input_mode"),
            "matrix_input_path": self._text("matrix_input_path"),
            "matrix_deepseek_api_key": self._text("matrix_deepseek_api_key"),
            "matrix_deepseek_model": self._text("matrix_deepseek_model"),
            "matrix_deepseek_base_url": self._text("matrix_deepseek_base_url"),
            "matrix_fail_fast": self._bool("matrix_fail_fast"),
            "summarizer_manifest": self._text("summarizer_manifest"),
            "summarizer_preset": self._text("summarizer_preset"),
            "summarizer_providers": self._text("summarizer_providers"),
            "summarizer_judge_provider": self._text("summarizer_judge_provider"),
            "summarizer_max_samples": self._text("summarizer_max_samples"),
            "qa_summary_xml": self._text("qa_summary_xml"),
            "qa_source_transcript": self._text("qa_source_transcript"),
            "qa_config_path": self._text("qa_config_path"),
            "qa_provider": self._text("qa_provider"),
            "qa_creator_provider": self._text("qa_creator_provider"),
            "qa_evaluator_provider": self._text("qa_evaluator_provider"),
            "qa_output_format": self._select_value("qa_output_format"),
            "qa_color": self._select_value("qa_color"),
            "qmsum_dataset_manifest": self._text("qmsum_dataset_manifest"),
            "qmsum_summarizer_provider": self._text("qmsum_summarizer_provider"),
            "qmsum_critic_provider": self._text("qmsum_critic_provider"),
            "qmsum_judge_provider": self._text("qmsum_judge_provider"),
            "qmsum_embedding_mode": self._select_value("qmsum_embedding_mode"),
            "qmsum_max_samples": self._text("qmsum_max_samples"),
            "qmsum_judge_repeats": self._text("qmsum_judge_repeats"),
            "qmsum_max_iterations": self._text("qmsum_max_iterations"),
            "qmsum_prepare_if_missing": self._bool("qmsum_prepare_if_missing"),
            "qmsum_force_download_dataset": self._bool("qmsum_force_download_dataset"),
            "sporc_dataset_manifest": self._text("sporc_dataset_manifest"),
            "sporc_dataset_variant": self._select_value("sporc_dataset_variant"),
            "sporc_summarizer_provider": self._text("sporc_summarizer_provider"),
            "sporc_critic_provider": self._text("sporc_critic_provider"),
            "sporc_judge_provider": self._text("sporc_judge_provider"),
            "sporc_embedding_mode": self._select_value("sporc_embedding_mode"),
            "sporc_duration_preset": self._text("sporc_duration_preset"),
            "sporc_max_samples": self._text("sporc_max_samples"),
            "sporc_max_iterations": self._text("sporc_max_iterations"),
            "sporc_transcript_only": self._bool("sporc_transcript_only"),
            "sporc_sweep_root": self._text("sporc_sweep_root"),
            "sporc_sweep_mode": self._select_value("sporc_sweep_mode"),
            "sporc_sweep_duration_presets": self._text("sporc_sweep_duration_presets"),
            "sporc_sweep_examples": self._text("sporc_sweep_examples"),
            "sporc_sweep_parallelism": self._text("sporc_sweep_parallelism"),
            "sporc_sweep_concurrency": self._text("sporc_sweep_concurrency"),
            "sporc_sweep_enable_voice_clone": self._bool(
                "sporc_sweep_enable_voice_clone"
            ),
            "diarization_manifest": self._text("diarization_manifest"),
            "diarization_providers": self._text("diarization_providers"),
            "diarization_device": self._select_value("diarization_device"),
            "diarization_max_samples": self._text("diarization_max_samples"),
            "diarization_collar": self._text("diarization_collar"),
            "diarization_skip_overlap": self._bool("diarization_skip_overlap"),
        }

    def _refresh_pipeline_visibility(self) -> None:
        mode = self._select_value("pipeline_mode")
        self.query_one("#pipeline_custom_stage_group").display = (
            self._select_value("pipeline_stage_strategy") == "custom"
        )
        self.query_one("#pipeline_audio_group").display = True
        self.query_one("#pipeline_additional_audio_group").display = mode in {
            "stage1_batch",
            "stage1_batch_duration_sweep",
        }
        self.query_one("#pipeline_duration_group").display = (
            mode == "stage1_batch_duration_sweep"
        )
        self.query_one("#pipeline_stage2_group").display = mode == "stage2_transcript"
        self.query_one("#pipeline_stage3_group").display = mode in {
            "stage3_summary",
            "stage4_interjector",
        }
        self.query_one("#pipeline_stage4_group").display = mode == "stage4_interjector"
        self.query_one("#pipeline_resume_group").display = mode == "resume_session"

    def _refresh_benchmark_visibility(self) -> None:
        kind = self._select_value("benchmark_kind")
        for benchmark_kind, _label in BENCHMARK_KIND_OPTIONS:
            self.query_one(f"#benchmark_group_{benchmark_kind}").display = (
                benchmark_kind == kind
            )

    @staticmethod
    def _is_pipeline_widget_id(widget_id: str) -> bool:
        return widget_id.startswith("pipeline_")

    @staticmethod
    def _is_benchmark_widget_id(widget_id: str) -> bool:
        return widget_id == "benchmark_kind" or widget_id.startswith(
            (
                "benchmark_",
                "matrix_",
                "summarizer_",
                "qa_",
                "qmsum_",
                "sporc_",
                "diarization_",
            )
        )

    def _refresh_previews(self) -> None:
        try:
            self._refresh_pipeline_preview()
            self._refresh_benchmark_preview()
            self._refresh_config_status()
        except NoMatches:
            return

    def _refresh_relevant_previews(self, widget_id: str) -> None:
        try:
            if not widget_id:
                self._refresh_pipeline_preview()
                self._refresh_benchmark_preview()
                return
            if self._is_pipeline_widget_id(widget_id):
                self._refresh_pipeline_preview()
            if self._is_benchmark_widget_id(widget_id):
                self._refresh_benchmark_preview()
        except NoMatches:
            return

    def _path_suggestion_specs(
        self,
    ) -> dict[str, tuple[str, tuple[str, ...]]]:
        try:
            matrix_input_mode = self._select_value("matrix_input_mode")
        except NoMatches:
            matrix_input_mode = "audio"
        matrix_suffixes = (
            (".json",)
            if matrix_input_mode == "transcript"
            else SUPPORTED_AUDIO_PATH_SUFFIXES
        )
        return {
            "pipeline_primary_audio_path": (
                "pipeline_primary_audio_suggestions",
                SUPPORTED_AUDIO_PATH_SUFFIXES,
            ),
            "pipeline_transcript_path": (
                "pipeline_transcript_suggestions",
                (".json",),
            ),
            "pipeline_stage2_audio_path": (
                "pipeline_stage2_audio_suggestions",
                SUPPORTED_AUDIO_PATH_SUFFIXES,
            ),
            "pipeline_summary_path": (
                "pipeline_summary_suggestions",
                (".xml",),
            ),
            "pipeline_voice_clone_manifest_path": (
                "pipeline_voice_clone_manifest_suggestions",
                (".json",),
            ),
            "matrix_input_path": (
                "matrix_input_suggestions",
                matrix_suffixes,
            ),
        }

    def _refresh_path_suggestions_for(self, input_id: str) -> None:
        suggestion_spec = self._path_suggestion_specs().get(input_id)
        if suggestion_spec is None:
            return
        list_id, suffixes = suggestion_spec
        try:
            suggestion_list = self.query_one(f"#{list_id}", OptionList)
        except NoMatches:
            self.call_after_refresh(self._refresh_path_suggestions_for, input_id)
            return
        suggestions = compute_path_suggestions(
            raw_value=self._text(input_id),
            workspace_root=self._workspace_root,
            allowed_suffixes=suffixes,
        )
        self._path_suggestion_values[list_id] = suggestions
        suggestion_list.set_options(list(suggestions))
        suggestion_list.display = bool(suggestions)
        suggestion_list.highlighted = 0 if suggestions else None

    def _refresh_all_path_suggestions(self) -> None:
        for input_id in self._path_suggestion_specs():
            self._refresh_path_suggestions_for(input_id)

    def _refresh_pipeline_preview(self) -> None:
        preview = self.query_one("#pipeline_preview", Static)
        try:
            selection = build_pipeline_selection(
                defaults=self._defaults.pipeline_defaults,
                values=self._pipeline_values(),
            )
        except FrontendValidationError as exc:
            preview.update(f"Pipeline validation: {exc}")
            return
        preview.update("\n".join(describe_pipeline_selection(selection)))

    def _refresh_benchmark_preview(self) -> None:
        preview = self.query_one("#benchmark_preview", Static)
        kind = cast(BenchmarkKind, self._select_value("benchmark_kind"))
        try:
            request = build_benchmark_request(
                kind=kind,
                values=self._benchmark_values(),
                uv_executable=self._uv_executable,
            )
        except FrontendValidationError as exc:
            preview.update(f"Benchmark validation: {exc}")
            return
        preview.update(
            "\n".join(
                [
                    *describe_benchmark_request(request),
                    "",
                    "Command:",
                    " ".join(request.command),
                ]
            )
        )

    def _refresh_config_status(self) -> None:
        status = self.query_one("#config_status", Static)
        dirty_suffix = "modified" if self._config_dirty else "saved"
        try:
            validate_config_text(self._text("config_editor"))
        except FrontendValidationError as exc:
            status.update(f"Config status: invalid ({dirty_suffix})\n\n{exc}")
            return
        status.update(
            "Config status: valid "
            f"({dirty_suffix})\n\nPress Ctrl+S or use Save config to persist changes."
        )

    def _validate_config_only(self) -> None:
        save_config_text(
            config_path=Path(self._defaults.config_path),
            config_text=self._text("config_editor"),
        )
        self._config_dirty = False

    def _set_select_options(
        self,
        *,
        widget_id: str,
        options: list[tuple[str, str]],
        desired_value: str,
    ) -> None:
        widget = self.query_one(f"#{widget_id}", Select)
        widget.set_options(options)
        legal_values = {value for _label, value in options}
        target_value = desired_value if desired_value in legal_values else options[0][1]
        widget.value = target_value

    def _refresh_profile_select_options(self) -> None:
        llm_options = self._as_select_options(list(self._defaults.llm_profiles))
        stage_profile_options = [
            ("Use the shared LLM", "default"),
            *llm_options,
        ]
        embedding_options = self._as_select_options(
            list(self._defaults.embedding_profiles)
        )
        self._set_select_options(
            widget_id="pipeline_shared_llm",
            options=llm_options,
            desired_value=self._defaults.pipeline_defaults.llm,
        )
        self._set_select_options(
            widget_id="pipeline_stage_llm_summarizer",
            options=stage_profile_options,
            desired_value=self._defaults.pipeline_defaults.stage_llm["summarizer"],
        )
        self._set_select_options(
            widget_id="pipeline_stage_llm_critic",
            options=stage_profile_options,
            desired_value=self._defaults.pipeline_defaults.stage_llm["critic"],
        )
        self._set_select_options(
            widget_id="pipeline_stage_llm_interjector",
            options=stage_profile_options,
            desired_value=self._defaults.pipeline_defaults.stage_llm["interjector"],
        )
        self._set_select_options(
            widget_id="pipeline_embedding",
            options=embedding_options,
            desired_value=self._defaults.pipeline_defaults.embedding,
        )

    def _apply_config_backed_defaults_to_widgets(self) -> None:
        self._refresh_profile_select_options()
        self.query_one(
            "#pipeline_voice_clone_enabled", Checkbox
        ).value = self._defaults.pipeline_defaults.voice_clone_enabled
        self.query_one(
            "#pipeline_interjector_enabled", Checkbox
        ).value = self._defaults.pipeline_defaults.interjector_enabled

    def _reload_defaults_from_disk(self) -> None:
        self._defaults = load_frontend_defaults(
            config_path=Path(self._defaults.config_path),
            workspace_root=self._workspace_root,
        )
        self._apply_config_backed_defaults_to_widgets()

    def _save_config(self) -> None:
        self._validate_config_only()
        self._reload_defaults_from_disk()
        self.notify(
            f"Saved config to {self._defaults.config_path}", severity="information"
        )
        self._refresh_all_path_suggestions()
        self._refresh_pipeline_preview()
        self._refresh_benchmark_preview()
        self._refresh_config_status()

    def _reload_config(self) -> None:
        self._reload_defaults_from_disk()
        editor = self.query_one("#config_editor", TextArea)
        editor.text = self._defaults.config_text
        self._config_dirty = False
        self._refresh_all_path_suggestions()
        self._refresh_pipeline_preview()
        self._refresh_benchmark_preview()
        self._refresh_config_status()
        self.notify("Reloaded config from disk.", severity="information")

    def _save_if_dirty(self) -> None:
        if self._config_dirty:
            self._save_config()

    def action_save_config(self) -> None:
        self._save_config()

    def action_quit_frontend(self) -> None:
        self.exit(FrontendResult(action="cancel", ui_mode="textual"))

    def on_select_changed(self, event: Select.Changed) -> None:
        self._refresh_pipeline_visibility()
        self._refresh_benchmark_visibility()
        widget_id = event.select.id or ""
        if widget_id == "matrix_input_mode":
            self._refresh_path_suggestions_for("matrix_input_path")
        self._refresh_relevant_previews(widget_id)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self._refresh_relevant_previews(event.checkbox.id or "")

    def on_input_changed(self, event: Input.Changed) -> None:
        widget_id = event.input.id or ""
        self._refresh_path_suggestions_for(widget_id)
        self._refresh_relevant_previews(widget_id)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "config_editor":
            self._config_dirty = True
            self._refresh_config_status()
            return
        self._refresh_relevant_previews(event.text_area.id or "")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        list_id = event.option_list.id or ""
        suggestion_values = self._path_suggestion_values.get(list_id, ())
        if not suggestion_values:
            return
        if not (0 <= event.option_index < len(suggestion_values)):
            return
        input_id = {
            "pipeline_primary_audio_suggestions": "pipeline_primary_audio_path",
            "pipeline_transcript_suggestions": "pipeline_transcript_path",
            "pipeline_stage2_audio_suggestions": "pipeline_stage2_audio_path",
            "pipeline_summary_suggestions": "pipeline_summary_path",
            "pipeline_voice_clone_manifest_suggestions": "pipeline_voice_clone_manifest_path",
            "matrix_input_suggestions": "matrix_input_path",
        }.get(list_id)
        if input_id is None:
            return
        self.query_one(f"#{input_id}", Input).value = suggestion_values[
            event.option_index
        ]
        self.query_one(f"#{input_id}", Input).focus()
        self._refresh_path_suggestions_for(input_id)
        self._refresh_relevant_previews(input_id)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        try:
            if button_id == "launch_pipeline":
                self._save_if_dirty()
                selection = build_pipeline_selection(
                    defaults=self._defaults.pipeline_defaults,
                    values=self._pipeline_values(),
                )
                self.exit(
                    FrontendResult(
                        action="pipeline",
                        pipeline_selection=selection,
                        ui_mode="textual",
                    )
                )
                return
            if button_id == "launch_benchmark":
                self._save_if_dirty()
                request = build_benchmark_request(
                    kind=cast(BenchmarkKind, self._select_value("benchmark_kind")),
                    values=self._benchmark_values(),
                    uv_executable=self._uv_executable,
                )
                self.exit(
                    FrontendResult(
                        action="benchmark",
                        benchmark_request=request,
                        ui_mode="textual",
                    )
                )
                return
            if button_id == "save_config_button":
                self._save_config()
                return
            if button_id == "reload_config_button":
                self._reload_config()
                return
            if button_id in {"cancel_pipeline", "cancel_benchmark", "close_frontend"}:
                self.exit(FrontendResult(action="cancel", ui_mode="textual"))
                return
        except FrontendValidationError as exc:
            self.notify(str(exc), severity="error")


def run_textual_frontend(
    *, defaults: FrontendDefaults, uv_executable: str
) -> FrontendResult:
    """Launch the Textual frontend and return the selected action."""
    app = SetupAndRunFrontendApp(defaults=defaults, uv_executable=uv_executable)
    result = app.run()
    if isinstance(result, FrontendResult):
        return result
    return FrontendResult(action="cancel", ui_mode="textual")
