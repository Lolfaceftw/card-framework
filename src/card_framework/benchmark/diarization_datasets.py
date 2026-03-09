"""Prepare public diarization datasets for local benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

import requests

from card_framework.benchmark.artifacts import utc_now_iso

AMI_AUDIO_BASE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
AMI_AUDIO_STREAM_SUFFIX = "Mix-Headset.wav"
AMI_SETUP_BASE_URL = "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main"
AMI_SUPPORTED_SUBSETS = frozenset({"train", "dev", "test"})


class DiarizationDatasetPreparationError(RuntimeError):
    """Raised when benchmark dataset preparation cannot proceed."""


class _ResponseProtocol(Protocol):
    """Describe the response interface used by dataset download helpers."""

    text: str

    def raise_for_status(self) -> None:
        """Raise an exception when the response is not successful."""

    def iter_content(self, chunk_size: int) -> Any:
        """Yield response chunks."""

    def __enter__(self) -> _ResponseProtocol:
        """Enter a context manager."""

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        """Exit a context manager."""


class _SessionProtocol(Protocol):
    """Describe the subset of ``requests.Session`` used by this module."""

    def get(
        self,
        url: str,
        *,
        stream: bool = False,
        timeout: tuple[float, float] | None = None,
    ) -> _ResponseProtocol:
        """Issue an HTTP GET request."""

    def close(self) -> None:
        """Release any session resources."""


@dataclass(frozen=True)
class AmiPreparedSample:
    """Store one prepared AMI diarization sample."""

    sample_id: str
    subset: str
    audio_path: str
    rttm_path: str
    uem_path: str
    num_speakers: int | None


def _resolve_path(path_value: Path) -> Path:
    """Resolve a path without requiring it to exist yet."""
    return path_value.expanduser().resolve()


def _download_file(
    *,
    session: _SessionProtocol,
    url: str,
    destination: Path,
    force: bool,
) -> Path:
    """Download one file unless a cached copy already exists."""
    destination = _resolve_path(destination)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with session.get(url, stream=True, timeout=(10.0, 300.0)) as response:
            response.raise_for_status()
            with NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=destination.parent,
            ) as handle:
                temp_path = Path(handle.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if temp_path is None:
            raise DiarizationDatasetPreparationError(
                f"Download failed before writing any data for {url}"
            )
        temp_path.replace(destination)
    except Exception as exc:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise DiarizationDatasetPreparationError(
            f"Failed to download '{url}' to '{destination}': {exc}"
        ) from exc
    return destination


def _load_meeting_ids(list_path: Path) -> list[str]:
    """Load ordered meeting identifiers from a prepared list file."""
    return [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _count_speakers_in_rttm(rttm_path: Path) -> int | None:
    """Count unique speakers in one RTTM file."""
    speakers: set[str] = set()
    for line in rttm_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            speakers.add(parts[7])
    return len(speakers) if speakers else None


def _meeting_list_url(subset: str) -> str:
    """Build the raw URL for one AMI setup meeting list."""
    return f"{AMI_SETUP_BASE_URL}/lists/{subset}.meetings.txt"


def _rttm_url(subset: str, meeting_id: str) -> str:
    """Build the raw URL for one AMI RTTM file."""
    return f"{AMI_SETUP_BASE_URL}/only_words/rttms/{subset}/{meeting_id}.rttm"


def _uem_url(subset: str, meeting_id: str) -> str:
    """Build the raw URL for one AMI UEM file."""
    return f"{AMI_SETUP_BASE_URL}/uems/{subset}/{meeting_id}.uem"


def _audio_url(meeting_id: str) -> str:
    """Build the public AMI audio URL for one meeting."""
    return (
        f"{AMI_AUDIO_BASE_URL}/{meeting_id}/audio/"
        f"{meeting_id}.{AMI_AUDIO_STREAM_SUFFIX}"
    )


def prepare_ami_manifest(
    *,
    output_path: Path,
    data_root: Path,
    subset: str = "test",
    num_samples: int = 0,
    force_download: bool = False,
    session: _SessionProtocol | None = None,
) -> dict[str, Any]:
    """Download public AMI diarization assets and write a manifest.

    This prep path uses public `Mix-Headset.wav` signals from the AMI corpus and
    RTTM/UEM/list files from `BUTSpeechFIT/AMI-diarization-setup`.
    """
    normalized_subset = subset.strip().lower()
    if normalized_subset not in AMI_SUPPORTED_SUBSETS:
        raise DiarizationDatasetPreparationError(
            "AMI subset must be one of: train, dev, test."
        )
    if num_samples < 0:
        raise DiarizationDatasetPreparationError("num_samples must be non-negative.")

    output_path = _resolve_path(output_path)
    data_root = _resolve_path(data_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    owns_session = session is None
    active_session = session or requests.Session()

    try:
        list_path = _download_file(
            session=active_session,
            url=_meeting_list_url(normalized_subset),
            destination=data_root / "lists" / f"{normalized_subset}.meetings.txt",
            force=force_download,
        )
        meeting_ids = _load_meeting_ids(list_path)
        if not meeting_ids:
            raise DiarizationDatasetPreparationError(
                f"No AMI meetings were found in {list_path}."
            )
        if num_samples > 0:
            meeting_ids = meeting_ids[:num_samples]

        prepared_samples: list[AmiPreparedSample] = []
        for meeting_id in meeting_ids:
            audio_path = _download_file(
                session=active_session,
                url=_audio_url(meeting_id),
                destination=(
                    data_root
                    / "audio"
                    / normalized_subset
                    / f"{meeting_id}.{AMI_AUDIO_STREAM_SUFFIX}"
                ),
                force=force_download,
            )
            rttm_path = _download_file(
                session=active_session,
                url=_rttm_url(normalized_subset, meeting_id),
                destination=data_root / "rttm" / normalized_subset / f"{meeting_id}.rttm",
                force=force_download,
            )
            uem_path = _download_file(
                session=active_session,
                url=_uem_url(normalized_subset, meeting_id),
                destination=data_root / "uem" / normalized_subset / f"{meeting_id}.uem",
                force=force_download,
            )
            prepared_samples.append(
                AmiPreparedSample(
                    sample_id=meeting_id,
                    subset=normalized_subset,
                    audio_path=str(audio_path),
                    rttm_path=str(rttm_path),
                    uem_path=str(uem_path),
                    num_speakers=_count_speakers_in_rttm(rttm_path),
                )
            )
    finally:
        if owns_session:
            active_session.close()

    manifest = {
        "manifest_version": "1",
        "created_at_utc": utc_now_iso(),
        "source": "ami",
        "subset": normalized_subset,
        "audio_stream": "mix-headset",
        "notes": (
            "Audio downloaded from the public AMI corpus; RTTM/UEM/list files "
            "downloaded from BUTSpeechFIT/AMI-diarization-setup only_words."
        ),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "dataset": "ami",
                "subset": sample.subset,
                "audio_filepath": sample.audio_path,
                "rttm_filepath": sample.rttm_path,
                "uem_filepath": sample.uem_path,
                "num_speakers": sample.num_speakers,
                "metadata": {
                    "audio_stream": "mix-headset",
                    "reference_setup": "BUTSpeechFIT/AMI-diarization-setup only_words",
                },
            }
            for sample in prepared_samples
        ],
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

