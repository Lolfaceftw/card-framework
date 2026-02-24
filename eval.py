import asyncio
import json
import logging
import sys
import time
from typing import Any, Dict, List

import hydra
import requests
from datasets import load_dataset
from omegaconf import DictConfig

from agents.critic import CriticExecutor
from agents.retrieval import InfoRetrievalExecutor
from agents.summarizer import SummarizerExecutor
from embeddings import TranscriptIndex
from llm_provider import EmbeddingProvider, LLMProvider
from logger_utils import configure_logger
from main import _build_a2a_app, _run_server_in_thread
from orchestrator import Orchestrator
from providers.logging_provider import LoggingLLMProvider
from ui import ui


def prepare_datasets(num_samples: int = 1) -> List[Dict[str, Any]]:
    """Download AMI and QMSum and format them into transcript segments."""
    eval_samples = []

    ui.print_system("Downloading QMSum dataset (test split)...")
    try:
        # Fallback to Yale-LILY/QMSum using streaming
        qmsum = load_dataset("Yale-LILY/QMSum", split="test", streaming=True)
        ui.print_status("QMSum stream connected.")

        qmsum_iter = iter(qmsum)
        for i in range(num_samples):
            try:
                example = next(qmsum_iter)
            except StopIteration:
                break

            # Yale-LILY/QMSum provides meetings. We'll reconstruct a segment list.
            segments = []
            for turn in example.get("meeting_transcripts", []):
                speaker = turn.get("speaker", "UNKNOWN")
                text = turn.get("utterance", "")
                segments.append({"speaker": speaker, "text": text})

            if segments:
                eval_samples.append(
                    {
                        "dataset": "QMSum",
                        "id": example.get("query_id", f"qmsum_{i}"),
                        "transcript": {"segments": segments},
                    }
                )

    except Exception as e:
        ui.print_error(f"Failed to load QMSum: {e}")

    ui.print_system("Downloading AMI dataset (test split)...")
    try:
        # Load AMI meeting corpus from HF using 'ihm' subset with streaming
        ami = load_dataset("edinburghcstr/ami", "ihm", split="test", streaming=True)
        ui.print_status("AMI stream connected.")

        ami_iter = iter(ami)
        for i in range(num_samples):
            try:
                # With AMI ihm, the dataset is usually utterances.
                # Let's read 50 consecutive utterances to build a fake meeting transcript block
                segments = []
                for j in range(50):
                    try:
                        row = next(ami_iter)
                        speaker = row.get("speaker_id", f"Speaker_{j % 4}")
                        text = row.get("text", "")
                        if text:
                            segments.append({"speaker": speaker, "text": text})
                    except StopIteration:
                        break

                if segments:
                    eval_samples.append(
                        {
                            "dataset": "AMI",
                            "id": f"ami_sample_{i}",
                            "transcript": {"segments": segments},
                        }
                    )
            except Exception:
                break
    except Exception as e:
        ui.print_error(f"Failed to load AMI: {e}")

    # Fallback to local transcript if datasets fail
    if not eval_samples:
        ui.print_error(
            "Failed to load HuggingFace datasets. Falling back to local summary.json..."
        )
        try:
            import os

            if os.path.exists("summary.json"):
                with open("summary.json", "r", encoding="utf-8-sig") as f:
                    local_data = json.load(f)
                    eval_samples.append(
                        {
                            "dataset": "Local_summary.json",
                            "id": "local_1",
                            "transcript": local_data,
                        }
                    )
            else:
                ui.print_error("summary.json not found.")
        except Exception as e:
            ui.print_error(f"Failed to load local summary.json: {e}")

    return eval_samples


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ui.print_system("Initializing evaluation script...")

    # ── 1. Configure Logging ──
    configure_logger(cfg.logging)
    if not cfg.logging.get("print_to_terminal", False):
        for logger_name in ["google", "google_genai", "httpx", "a2a", "uvicorn"]:
            logging.getLogger(logger_name).propagate = False

    # ── 2. Instantiate providers ──
    ui.print_system("Instantiating providers...")
    llm: LLMProvider = hydra.utils.instantiate(cfg.llm)
    if cfg.get("logging", {}).get("enabled", False):
        llm = LoggingLLMProvider(inner_provider=llm)

    embedding: EmbeddingProvider = hydra.utils.instantiate(cfg.embedding)
    transcript_index = TranscriptIndex(embedding_provider=embedding)

    # ── 3. Start A2A servers ──
    # We will spin these up in daemon threads exactly as main.py does.
    retrieval_port = cfg.ports.retrieval
    summarizer_port = cfg.ports.summarizer
    critic_port = cfg.ports.critic

    ui.print_system(
        f"Starting A2A servers (R:{retrieval_port} S:{summarizer_port} C:{critic_port})..."
    )

    retrieval_app = _build_a2a_app(
        "InfoRetrieval",
        "Retrieve segments",
        retrieval_port,
        InfoRetrievalExecutor(transcript_index),
    )
    _run_server_in_thread("retrieval-a2a", retrieval_app, retrieval_port)

    summarizer_app = _build_a2a_app(
        "Summarizer",
        "Summarize",
        summarizer_port,
        SummarizerExecutor(
            llm=llm,
            retrieval_port=retrieval_port,
            max_tool_turns=cfg.get("agents", {})
            .get("summarizer", {})
            .get("max_tool_turns", 3),
        ),
    )
    _run_server_in_thread("summarizer-a2a", summarizer_app, summarizer_port)

    critic_app = _build_a2a_app(
        "Critic",
        "Evaluate",
        critic_port,
        CriticExecutor(
            llm=llm,
            max_tool_turns=cfg.get("agents", {})
            .get("critic", {})
            .get("max_tool_turns", 5),
            retrieval_port=retrieval_port,
        ),
    )
    _run_server_in_thread("critic-a2a", critic_app, critic_port)

    ui.print_system("Waiting for A2A servers to start...")
    time.sleep(2)

    for name, port in [
        ("InfoRetrieval", retrieval_port),
        ("Summarizer", summarizer_port),
        ("Critic", critic_port),
    ]:
        try:
            r = requests.get(
                f"http://127.0.0.1:{port}/.well-known/agent.json", timeout=5
            )
            r.raise_for_status()
        except Exception as e:
            ui.print_error(f"[ERR] {name} server not responding: {e}")
            sys.exit(1)

    # ── 4. Prepare Evaluation Data ──
    num_samples = 1  # We evaluate 1 sample per dataset to limit time
    samples = prepare_datasets(num_samples=num_samples)

    if not samples:
        ui.print_error("No samples loaded. Aborting evaluation.")
        sys.exit(1)

    # ── 5. Run Orchestration Loop for Each Sample ──
    orchestrator = Orchestrator(
        retrieval_port=retrieval_port,
        summarizer_port=summarizer_port,
        critic_port=critic_port,
        timeouts=dict(cfg.orchestrator.get("timeouts", {})),
    )

    eval_report = {"overall_samples": len(samples), "results": []}

    async def run_evals():
        for i, sample in enumerate(samples):
            dataset_name = sample["dataset"]
            sample_id = sample["id"]
            transcript = sample["transcript"]

            ui.print_status(
                f"\nEvaluating Sample {i + 1}/{len(samples)}: [{dataset_name}] ID: {sample_id}"
            )
            ui.print_system(f"Segments: {len(transcript.get('segments', []))}")

            # Override parameters if needed for testing bounds
            min_w = cfg.orchestrator.min_words
            max_w = cfg.orchestrator.max_words
            max_iter = cfg.orchestrator.max_iterations

            index_count = await orchestrator.index_transcript(transcript)
            ui.print_system(f"Indexed {index_count} segments for {sample_id}.")

            t0 = time.time()
            # To track iterations, we slightly hook into the loop or just read the console out.
            # For accurate tracking, since `run_loop` doesn't return iteration count cleanly,
            # we just call it and assume if it returns a draft, it passed.
            result = await orchestrator.run_loop(min_w, max_w, max_iterations=max_iter)
            t1 = time.time()

            passed = bool(result)
            # In a true rigorous eval, you might modify `Orchestrator.run_loop` to return
            # more metadata (iterations, feedback traces, final word count).
            # For now, we estimate based on whether it returned a draft.

            word_count = len(result.split()) if result else 0

            eval_report["results"].append(
                {
                    "dataset": dataset_name,
                    "id": sample_id,
                    "passed": passed,
                    "word_count": word_count,
                    "duration_seconds": round(t1 - t0, 2),
                    "draft": result,
                }
            )

            ui.print_status(f"Finished evaluating {sample_id}. Passed: {passed}.")

    asyncio.run(run_evals())

    # ── 6. Save Report ──
    report_file = "eval_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    ui.print_status(f"\nEvaluation complete. Report saved to {report_file}.")

    passes = sum(1 for r in eval_report["results"] if r["passed"])
    ui.print_status(f"Overall Pass Rate: {passes}/{len(samples)}")


if __name__ == "__main__":
    main()
