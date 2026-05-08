"""Analyze partial human-eval progress and Gemini judge alignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from card_framework.benchmark.reference_free.conversation_judge import (
    ConversationSummaryJudge,
)
from card_framework.providers.google_genai_provider import GoogleGenAIProvider
from card_framework.shared.env import load_repo_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
RESPONSES_CSV = (
    REPO_ROOT
    / "research"
    / "Podcast Summary Evaluation (Responses) - Form Responses 1.csv"
)
ASSIGNMENTS_CSV = (
    REPO_ROOT / "research" / "human_eval_analysis" / "participant_assignments.csv"
)
OUTPUT_DIR = REPO_ROOT / "research" / "human_eval_analysis"
DOWNLOAD_ROOT = REPO_ROOT / "human_eval_gdrive_artifacts"

LIKERT_TO_NORM = {
    1: 0.0,
    2: 0.25,
    3: 0.5,
    4: 0.75,
    5: 1.0,
}

DIMENSION_PREFIXES = {
    "Factualness": "Factualness",
    "Naturalness": "Naturalness",
    "Speaker Grammar Similarity": "Speaker Grammar Similarity",
    "Vocal Delivery": "Vocal Delivery",
    "Conversational Flow": "Conversational Flow",
    "Listener Experience": "Listener Experience",
}

GENERATOR_RUN_SUMMARIES = (
    REPO_ROOT
    / "artifacts"
    / "sporc_human_eval_stage2_sweeps"
    / "full_20260321T221039Z"
    / "run_summary.json",
    REPO_ROOT
    / "artifacts"
    / "sporc_human_eval_stage2_sweeps"
    / "deepseek_reasoner_remaining_20260322T0915Z"
    / "remaining_short"
    / "run_summary.json",
    REPO_ROOT
    / "artifacts"
    / "sporc_human_eval_stage2_sweeps"
    / "deepseek_reasoner_remaining_20260322T0915Z"
    / "remaining_5m"
    / "run_summary.json",
)


@dataclass(frozen=True)
class Assignment:
    participant_id: str
    participant_email: str
    assigned_podcast: str
    assigned_duration: str
    folder_link: str


@dataclass(frozen=True)
class ArtifactRecord:
    podcast_id: str
    duration: str
    episode_index: int
    episode_title: str
    generator_profile: str
    generator_model: str
    output_root: Path
    transcript_path: Path
    summary_path: Path
    final_audio_path: Path | None
    folder_link: str

    @property
    def artifact_key(self) -> str:
        return f"{self.podcast_id}_{self.duration}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-csv", type=Path, default=RESPONSES_CSV)
    parser.add_argument("--assignments-csv", type=Path, default=ASSIGNMENTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--download-root", type=Path, default=DOWNLOAD_ROOT)
    parser.add_argument("--download-artifacts", action="store_true")
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Reuse cached Gemini scores only; do not invoke the API.",
    )
    return parser.parse_args()


def _load_assignments(path: Path) -> dict[str, Assignment]:
    assignments: dict[str, Assignment] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            participant_id = str(row["participant_id"]).strip().zfill(2)
            assignments[participant_id] = Assignment(
                participant_id=participant_id,
                participant_email=str(row["participant_email"]).strip(),
                assigned_podcast=str(row["assigned_podcast"]).strip(),
                assigned_duration=str(row["assigned_duration"]).strip(),
                folder_link=str(row["folder_link"]).strip(),
            )
    return assignments


def _load_local_artifacts(
    assignments: dict[str, Assignment],
) -> dict[tuple[str, str], ArtifactRecord]:
    folder_links = {
        (
            assignment.assigned_podcast,
            assignment.assigned_duration,
        ): assignment.folder_link
        for assignment in assignments.values()
    }
    artifacts: dict[tuple[str, str], ArtifactRecord] = {}
    for summary_path in GENERATOR_RUN_SUMMARIES:
        payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        generator_profile = str(
            payload.get("llm_profile") or payload.get("gemini_model") or "unknown"
        )
        generator_model = str(
            payload.get("llm_model") or payload.get("gemini_model") or "unknown"
        )
        for run in payload.get("runs", []):
            if str(run.get("status", "")).strip().lower() != "completed":
                continue
            episode_index = int(run["episode_index"])
            duration = str(run["duration_preset"]).strip()
            podcast_id = f"Podcast {episode_index:02d}"
            artifacts[(podcast_id, duration)] = ArtifactRecord(
                podcast_id=podcast_id,
                duration=duration,
                episode_index=episode_index,
                episode_title=str(run["episode_title"]).strip(),
                generator_profile=generator_profile,
                generator_model=generator_model,
                output_root=Path(run["output_root"]),
                transcript_path=Path(run["artifacts"]["transcript_path"]),
                summary_path=Path(run["artifacts"]["summary_xml_path"]),
                final_audio_path=Path(run["artifacts"]["final_audio_path"])
                if run.get("artifacts", {}).get("final_audio_path")
                else None,
                folder_link=folder_links.get((podcast_id, duration), ""),
            )
    return artifacts


def _extract_numeric_rating(raw_value: str) -> int | None:
    stripped = str(raw_value).strip()
    if not stripped:
        return None
    head = stripped.split(" ", 1)[0]
    if head.isdigit():
        return int(head)
    return None


def _group_dimension_scores(
    row: dict[str, str],
) -> tuple[dict[str, float], dict[str, float]]:
    normalized: dict[str, float] = {}
    raw_means: dict[str, float] = {}
    for dimension, prefix in DIMENSION_PREFIXES.items():
        values: list[int] = []
        for column_name, cell in row.items():
            if column_name.startswith(prefix):
                numeric = _extract_numeric_rating(cell)
                if numeric is not None:
                    values.append(numeric)
        if not values:
            continue
        raw_means[dimension] = sum(values) / len(values)
        normalized[dimension] = sum(LIKERT_TO_NORM[value] for value in values) / len(
            values
        )
    return normalized, raw_means


def _load_responses(
    path: Path,
    assignments: dict[str, Assignment],
) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            participant_id = (
                str(row["Participant ID (from the email)"]).strip().zfill(2)
            )
            assignment = assignments.get(participant_id)
            normalized, raw_means = _group_dimension_scores(row)
            responses.append(
                {
                    "participant_id": participant_id,
                    "participant_email": str(row["Email Address"]).strip(),
                    "timestamp": str(row["Timestamp"]).strip(),
                    "assignment": assignment,
                    "normalized_scores": normalized,
                    "raw_means": raw_means,
                    "comment": str(
                        row.get(
                            "Provide any additional qualitative remarks or observations regarding the podcast summary.",
                            "",
                        )
                    ).strip(),
                }
            )
    return responses


def _prepare_download_layout(
    download_root: Path,
    artifacts: dict[tuple[str, str], ArtifactRecord],
    *,
    download_artifacts: bool,
) -> dict[str, Any]:
    gdown = None
    if download_artifacts:
        import importlib

        gdown = importlib.import_module("gdown")

    download_root.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    manifest_rows = []
    for artifact in sorted(
        artifacts.values(), key=lambda item: (item.podcast_id, item.duration)
    ):
        target_dir = (
            download_root / artifact.podcast_id.replace(" ", "_") / artifact.duration
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows.append(
            {
                "artifact_key": artifact.artifact_key,
                "podcast_id": artifact.podcast_id,
                "duration": artifact.duration,
                "folder_link": artifact.folder_link,
                "generator_model": artifact.generator_model,
                "local_output_root": str(artifact.output_root),
            }
        )
        if not download_artifacts or not artifact.folder_link:
            continue
        if gdown is None:
            raise RuntimeError(
                "gdown is required when --download-artifacts is enabled."
            )
        try:
            gdown.download_folder(
                url=artifact.folder_link,
                output=str(target_dir),
                quiet=True,
                use_cookies=False,
            )
            attempts.append(
                {"artifact_key": artifact.artifact_key, "status": "downloaded"}
            )
        except Exception as exc:  # pragma: no cover - depends on remote access
            attempts.append(
                {
                    "artifact_key": artifact.artifact_key,
                    "status": "blocked",
                    "error": str(exc),
                }
            )
    manifest_path = download_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    readme_lines = [
        "# Human evaluation Google Drive artifact layout",
        "",
        "This folder mirrors the podcast/duration assignment structure.",
        "The local workspace already contains the matching generated artifacts under `artifacts/sporc_human_eval_stage2_sweeps/`.",
        "",
    ]
    if attempts:
        blocked = [item for item in attempts if item["status"] != "downloaded"]
        if blocked:
            readme_lines.extend(
                [
                    "## Download status",
                    "",
                    "Some Google Drive folders were not downloadable from this environment.",
                    "The captured blocker details are in `download_status.json`.",
                    "",
                ]
            )
    (download_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    (download_root / "download_status.json").write_text(
        json.dumps(attempts, indent=2), encoding="utf-8"
    )
    return {
        "manifest_path": str(manifest_path),
        "download_status_path": str(download_root / "download_status.json"),
        "attempts": attempts,
    }


def _transcript_source_text(transcript_payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for segment in transcript_payload.get("segments", []):
        if not isinstance(segment, dict):
            continue
        speaker = str(segment.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
        text = str(segment.get("text", "")).strip()
        if text:
            lines.append(f"[{speaker}]: {text}")
    return "\n".join(lines)


def _canonical_speaker_label(label: str, alias_map: dict[str, str]) -> str:
    cleaned = label.strip()
    if not cleaned:
        return "UNKNOWN"
    direct = alias_map.get(cleaned)
    if direct:
        return direct
    cleaned_casefold = cleaned.casefold()
    for alias_key, canonical in alias_map.items():
        if alias_key.casefold() == cleaned_casefold:
            return canonical
    return cleaned


def _transcript_per_speaker_messages(
    transcript_payload: dict[str, Any], alias_map: dict[str, str]
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for segment in transcript_payload.get("segments", []):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        speaker = _canonical_speaker_label(
            str(segment.get("speaker", "UNKNOWN")), alias_map
        )
        grouped[speaker].append(text)
    return dict(grouped)


def _summary_per_speaker_messages(
    summary_xml: str, alias_map: dict[str, str]
) -> dict[str, list[str]]:
    import re

    grouped: dict[str, list[str]] = defaultdict(list)
    pattern = re.compile(
        r"<(?P<speaker>[^\s>/]+)(?:\s+[^>]*)?>(?P<text>.*?)</[^>]+>", re.DOTALL
    )
    for match in pattern.finditer(summary_xml):
        speaker = _canonical_speaker_label(match.group("speaker"), alias_map)
        text = re.sub(r"\s+", " ", match.group("text")).strip()
        if text:
            grouped[speaker].append(text)
    return dict(grouped)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        avg_rank = (cursor + 1 + end) / 2.0
        for original_index, _ in indexed[cursor:end]:
            ranks[original_index] = avg_rank
        cursor = end
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _artifact_mean(rows: list[dict[str, Any]], dimension: str) -> float | None:
    values = [
        row["normalized_scores"][dimension]
        for row in rows
        if dimension in row["normalized_scores"]
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _run_gemini_judge(
    artifacts: dict[tuple[str, str], ArtifactRecord],
    cache_path: Path,
    *,
    skip_gemini: bool,
) -> dict[str, Any]:
    cached: dict[str, Any] = {}
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))

    missing = [
        artifact
        for artifact in artifacts.values()
        if artifact.artifact_key not in cached
    ]
    if missing and skip_gemini:
        missing_keys = ", ".join(sorted(item.artifact_key for item in missing))
        raise RuntimeError(f"Missing cached Gemini scores for: {missing_keys}")

    if missing:
        load_repo_dotenv(REPO_ROOT)
        api_key = str(os.environ.get("GEMINI_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not available in the environment.")
        judge = ConversationSummaryJudge(
            judge_llm=GoogleGenAIProvider(
                api_key=api_key,
                model="gemini-3.1-pro-preview",
                enable_thinking=True,
                request_timeout_seconds=120.0,
            )
        )
        for artifact in sorted(
            missing, key=lambda item: (item.episode_index, item.duration)
        ):
            transcript_payload = json.loads(
                artifact.transcript_path.read_text(encoding="utf-8-sig")
            )
            summary_xml = artifact.summary_path.read_text(encoding="utf-8")
            metadata = transcript_payload.get("metadata", {})
            alias_map = (
                metadata.get("speaker_alias_map", {})
                if isinstance(metadata, dict)
                else {}
            )
            speaker_notes = (
                metadata.get("speaker_notes", []) if isinstance(metadata, dict) else []
            )
            result = judge.evaluate(
                source_text=_transcript_source_text(transcript_payload),
                summary_xml=summary_xml,
                transcript_per_speaker_messages=_transcript_per_speaker_messages(
                    transcript_payload, alias_map if isinstance(alias_map, dict) else {}
                ),
                summary_per_speaker_messages=_summary_per_speaker_messages(
                    summary_xml, alias_map if isinstance(alias_map, dict) else {}
                ),
                speaker_alias_map=alias_map if isinstance(alias_map, dict) else {},
                speaker_notes=[str(item) for item in speaker_notes]
                if isinstance(speaker_notes, list)
                else [],
            )
            if result.status != "ok" or result.scores is None:
                raise RuntimeError(
                    f"Gemini judge failed for {artifact.artifact_key}: {result.error_message}"
                )
            cached[artifact.artifact_key] = {
                "podcast_id": artifact.podcast_id,
                "duration": artifact.duration,
                "generator_model": artifact.generator_model,
                "scores": result.scores,
                "composite": round(
                    sum(
                        float(result.scores[name])
                        for name in (
                            "Factualness",
                            "Naturalness",
                            "Speaker Grammar Similarity",
                        )
                    )
                    / 3.0,
                    4,
                ),
            }
        cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
    return cached


def _build_correlation_report(
    responses: list[dict[str, Any]],
    gemini_scores: dict[str, Any],
) -> dict[str, Any]:
    matched_responses = [
        response
        for response in responses
        if response["assignment"] is not None
        and f"{response['assignment'].assigned_podcast}_{response['assignment'].assigned_duration}"
        in gemini_scores
    ]
    dimensions = [
        "Factualness",
        "Naturalness",
        "Speaker Grammar Similarity",
    ]
    response_level: dict[str, Any] = {}
    for dimension in dimensions:
        human_values = []
        gemini_values = []
        for response in matched_responses:
            if dimension not in response["normalized_scores"]:
                continue
            artifact_key = f"{response['assignment'].assigned_podcast}_{response['assignment'].assigned_duration}"
            human_values.append(float(response["normalized_scores"][dimension]))
            gemini_values.append(
                float(gemini_scores[artifact_key]["scores"][dimension])
            )
        response_level[dimension] = {
            "n": len(human_values),
            "pearson": _round_or_none(_pearson(human_values, gemini_values)),
            "spearman": _round_or_none(_spearman(human_values, gemini_values)),
            "human_mean": round(sum(human_values) / len(human_values), 4)
            if human_values
            else None,
            "gemini_mean": round(sum(gemini_values) / len(gemini_values), 4)
            if gemini_values
            else None,
        }
    human_composite = []
    gemini_composite = []
    for response in matched_responses:
        per_dim = [response["normalized_scores"].get(name) for name in dimensions]
        if any(value is None for value in per_dim):
            continue
        artifact_key = f"{response['assignment'].assigned_podcast}_{response['assignment'].assigned_duration}"
        human_composite.append(sum(float(value) for value in per_dim) / 3.0)
        gemini_composite.append(float(gemini_scores[artifact_key]["composite"]))
    response_level["Composite"] = {
        "n": len(human_composite),
        "pearson": _round_or_none(_pearson(human_composite, gemini_composite)),
        "spearman": _round_or_none(_spearman(human_composite, gemini_composite)),
        "human_mean": round(sum(human_composite) / len(human_composite), 4)
        if human_composite
        else None,
        "gemini_mean": round(sum(gemini_composite) / len(gemini_composite), 4)
        if gemini_composite
        else None,
    }

    grouped_by_artifact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in matched_responses:
        artifact_key = f"{response['assignment'].assigned_podcast}_{response['assignment'].assigned_duration}"
        grouped_by_artifact[artifact_key].append(response)
    artifact_level: dict[str, Any] = {}
    for dimension in dimensions:
        human_values = []
        gemini_values = []
        for artifact_key, rows in sorted(grouped_by_artifact.items()):
            mean_value = _artifact_mean(rows, dimension)
            if mean_value is None:
                continue
            human_values.append(mean_value)
            gemini_values.append(
                float(gemini_scores[artifact_key]["scores"][dimension])
            )
        artifact_level[dimension] = {
            "n": len(human_values),
            "pearson": _round_or_none(_pearson(human_values, gemini_values)),
            "spearman": _round_or_none(_spearman(human_values, gemini_values)),
            "human_mean": round(sum(human_values) / len(human_values), 4)
            if human_values
            else None,
            "gemini_mean": round(sum(gemini_values) / len(gemini_values), 4)
            if gemini_values
            else None,
        }
    artifact_human_composite = []
    artifact_gemini_composite = []
    for artifact_key, rows in sorted(grouped_by_artifact.items()):
        dimension_means = [_artifact_mean(rows, dimension) for dimension in dimensions]
        if any(value is None for value in dimension_means):
            continue
        safe_dimension_means = [
            float(value) for value in dimension_means if value is not None
        ]
        artifact_human_composite.append(sum(safe_dimension_means) / 3.0)
        artifact_gemini_composite.append(
            float(gemini_scores[artifact_key]["composite"])
        )
    artifact_level["Composite"] = {
        "n": len(artifact_human_composite),
        "pearson": _round_or_none(
            _pearson(artifact_human_composite, artifact_gemini_composite)
        ),
        "spearman": _round_or_none(
            _spearman(artifact_human_composite, artifact_gemini_composite)
        ),
        "human_mean": round(
            sum(artifact_human_composite) / len(artifact_human_composite), 4
        )
        if artifact_human_composite
        else None,
        "gemini_mean": round(
            sum(artifact_gemini_composite) / len(artifact_gemini_composite), 4
        )
        if artifact_gemini_composite
        else None,
    }
    return {
        "matched_response_count": len(matched_responses),
        "artifact_count_with_responses": len(grouped_by_artifact),
        "response_level": response_level,
        "artifact_level": artifact_level,
    }


def _write_response_level_csv(
    path: Path,
    responses: list[dict[str, Any]],
    gemini_scores: dict[str, Any],
) -> None:
    fieldnames = [
        "participant_id",
        "participant_email",
        "podcast_id",
        "duration",
        "factualness_human",
        "naturalness_human",
        "speaker_human",
        "composite_human",
        "factualness_gemini",
        "naturalness_gemini",
        "speaker_gemini",
        "composite_gemini",
        "vocal_delivery_human",
        "conversational_flow_human",
        "listener_experience_human",
        "comment",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for response in responses:
            assignment = response["assignment"]
            if assignment is None:
                continue
            artifact_key = (
                f"{assignment.assigned_podcast}_{assignment.assigned_duration}"
            )
            gemini = gemini_scores.get(artifact_key)
            if not gemini:
                continue
            factualness = response["normalized_scores"].get("Factualness")
            naturalness = response["normalized_scores"].get("Naturalness")
            speaker = response["normalized_scores"].get("Speaker Grammar Similarity")
            composite_human = (
                round(
                    (float(factualness) + float(naturalness) + float(speaker)) / 3.0, 4
                )
                if factualness is not None
                and naturalness is not None
                and speaker is not None
                else None
            )
            writer.writerow(
                {
                    "participant_id": response["participant_id"],
                    "participant_email": response["participant_email"],
                    "podcast_id": assignment.assigned_podcast,
                    "duration": assignment.assigned_duration,
                    "factualness_human": factualness,
                    "naturalness_human": naturalness,
                    "speaker_human": speaker,
                    "composite_human": composite_human,
                    "factualness_gemini": gemini["scores"]["Factualness"],
                    "naturalness_gemini": gemini["scores"]["Naturalness"],
                    "speaker_gemini": gemini["scores"]["Speaker Grammar Similarity"],
                    "composite_gemini": gemini["composite"],
                    "vocal_delivery_human": response["normalized_scores"].get(
                        "Vocal Delivery"
                    ),
                    "conversational_flow_human": response["normalized_scores"].get(
                        "Conversational Flow"
                    ),
                    "listener_experience_human": response["normalized_scores"].get(
                        "Listener Experience"
                    ),
                    "comment": response["comment"],
                }
            )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments = _load_assignments(args.assignments_csv)
    artifacts = _load_local_artifacts(assignments)
    responses = _load_responses(args.responses_csv, assignments)
    download_status = _prepare_download_layout(
        args.download_root,
        artifacts,
        download_artifacts=args.download_artifacts,
    )
    artifact_manifest_path = args.output_dir / "artifact_manifest.json"
    artifact_manifest_path.write_text(
        json.dumps(
            {
                artifact.artifact_key: {
                    "podcast_id": artifact.podcast_id,
                    "duration": artifact.duration,
                    "episode_title": artifact.episode_title,
                    "generator_profile": artifact.generator_profile,
                    "generator_model": artifact.generator_model,
                    "output_root": str(artifact.output_root),
                    "transcript_path": str(artifact.transcript_path),
                    "summary_path": str(artifact.summary_path),
                    "final_audio_path": str(artifact.final_audio_path)
                    if artifact.final_audio_path
                    else None,
                    "folder_link": artifact.folder_link,
                }
                for artifact in sorted(
                    artifacts.values(),
                    key=lambda item: (item.episode_index, item.duration),
                )
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    gemini_cache_path = args.output_dir / "gemini_judge_scores.json"
    gemini_scores = _run_gemini_judge(
        artifacts,
        gemini_cache_path,
        skip_gemini=args.skip_gemini,
    )
    correlation_report = _build_correlation_report(responses, gemini_scores)
    audio_quality_summary: dict[str, Any] = {}
    for dimension in (
        "Vocal Delivery",
        "Conversational Flow",
        "Listener Experience",
    ):
        values = [
            float(response["normalized_scores"][dimension])
            for response in responses
            if dimension in response["normalized_scores"]
        ]
        audio_quality_summary[dimension] = {
            "n": len(values),
            "mean": round(sum(values) / len(values), 4) if values else None,
        }
    report = {
        "responses_csv": str(args.responses_csv),
        "assignments_csv": str(args.assignments_csv),
        "artifact_manifest_path": str(artifact_manifest_path),
        "gemini_judge_scores_path": str(gemini_cache_path),
        "download_layout": download_status,
        "total_assignment_count": len(assignments),
        "response_count": len(responses),
        "answered_participant_ids": sorted(
            response["participant_id"] for response in responses
        ),
        "pending_participant_ids": sorted(
            participant_id
            for participant_id in assignments
            if participant_id
            not in {response["participant_id"] for response in responses}
        ),
        "correlation": correlation_report,
        "audio_quality_summary": audio_quality_summary,
    }
    (args.output_dir / "correlation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_response_level_csv(
        args.output_dir / "response_level_alignment.csv", responses, gemini_scores
    )
    print(json.dumps(report["correlation"], indent=2))


if __name__ == "__main__":
    main()
