"""
LLM-enhanced multi-speaker podcast synthesis with advanced trigger detection.

Uses Mistral 8B (HF 4-bit) to detect trigger phrases and generate
contextual interjections with cross-fading between segments.
"""

import json
import os
import random
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import torch
from pydub import AudioSegment

from indextts.infer_v2 import IndexTTS2
from tools.podcast.hf_backchannel import MODEL_ID, ensure_model, extract_first_json, generate_text
from tools.podcast.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Initialize Index-TTS-2
_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    device=_DEVICE,
    use_fp16=_DEVICE.startswith("cuda"),
    use_cuda_kernel=_DEVICE.startswith("cuda"),
)


def ensure_mistral_loaded() -> bool:
    """Ensure HF Mistral 8B is available."""
    return ensure_model()


def detect_trigger_with_llm(text: str) -> list:
    """Use Mistral 8B to identify the best trigger point for an interjection.

    Args:
        text: Speaker text to analyze.

    Returns:
        List containing trigger data dict, or empty list on failure.
    """
    prompt = f"""
You are an expert conversation analyst. Your task is to find the single best, high-impact trigger for a listener to make a natural interjection. Analyze the speaker's text below.

The text is:
"{text}"

Focus on these specific categories for triggers:
- "question": A direct question. (e.g., "what do you think?", "how did that happen")
- "statement": A strong, surprising, or emotional declaration. (e.g., "it was unbelievable", "absolutely shocking")
- "problem": A word or phrase that introduces a point of concern. (e.g., "the main issue is", "I'm worried that")
- "agreement_seek": A phrase that explicitly seeks validation from the listener. (e.g., "isn't it?", "right?")

**AVOID**: Common filler words like "like", "so", "um", "you know" unless they are part of a larger, more meaningful trigger phrase. Choose the most significant point that invites a reaction.

Respond with ONLY a single JSON object with the keys "trigger_word", "char_pos", and "category".

---
EXAMPLE 1
Text: "We thought the launch would be simple, but the main issue was the server unexpectedly crashing."
{{
  "trigger_word": "issue",
  "char_pos": 46,
  "category": "problem"
}}
---
EXAMPLE 2
Text: "And the final result was, to be honest, absolutely amazing."
{{
  "trigger_word": "amazing",
  "char_pos": 51,
  "category": "statement"
}}
---
EXAMPLE 3
Text: "That's the plan anyway, but it's a bit risky, don't you think?"
{{
  "trigger_word": "don't you think?",
  "char_pos": 49,
  "category": "agreement_seek"
}}
---

Now, analyze the provided text and give your JSON response.
"""

    try:
        result_text = generate_text(
            [
                {"role": "system", "content": "You are an expert conversation analyst."},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=80,
            temperature=0.3,
            top_p=0.9,
        )
        if result_text:
            logger.debug(f"LLM trigger response: {result_text}")
            trigger_data = extract_first_json(result_text)
            if trigger_data and "trigger_word" in trigger_data and "char_pos" in trigger_data:
                trigger_data["pos_percent"] = trigger_data["char_pos"] / len(text)
                return [trigger_data]
    except Exception as e:
        logger.warning(f"LLM trigger detection error: {e}")

    return []


def calculate_seconds_per_word_from_audio(text: str, audio_path: str) -> float:
    """Calculate actual seconds-per-word directly from synthesized audio.

    Args:
        text: Text that was synthesized.
        audio_path: Path to the audio file.

    Returns:
        Seconds per word based on audio duration.
    """
    audio = AudioSegment.from_wav(audio_path)
    audio_duration_seconds = len(audio) / 1000.0
    word_count = len(text.split())
    return audio_duration_seconds / word_count


def calculate_interjection_timing_and_trigger(main_text: str, main_audio_path: str,
                                              seconds_per_word: float) -> dict:
    """Calculate optimal interjection timing based on trigger words.

    Args:
        main_text: Main speaker's text.
        main_audio_path: Path to main speaker's audio.
        seconds_per_word: Calibrated speech rate.

    Returns:
        Dict with timing info, or None if no trigger found.
    """
    main_audio = AudioSegment.from_wav(main_audio_path)
    audio_duration_ms = len(main_audio)

    logger.debug("Asking LLM to find a trigger point...")
    triggers = detect_trigger_with_llm(main_text)

    if not triggers:
        return None

    selected_trigger = random.choice(triggers)
    trigger_pos_percent = selected_trigger["pos_percent"]

    trigger_time_ms = int(audio_duration_ms * trigger_pos_percent)
    reaction_delay = random.randint(300, 800)
    interjection_position = trigger_time_ms + reaction_delay

    interjection_position = max(0, min(interjection_position, audio_duration_ms - 500))

    return {
        "position_ms": interjection_position,
        "trigger_word": selected_trigger["trigger_word"],
        "trigger_category": selected_trigger["category"],
        "trigger_position_percent": trigger_pos_percent,
        "trigger_position_ms": trigger_time_ms,
        "reaction_delay_ms": reaction_delay,
        "audio_duration_ms": audio_duration_ms
    }


def generate_interjection_llm(main_speaker_text: str, speaker_name: str) -> str:
    """Generate interjection using Mistral 8B (HF 4-bit).

    Args:
        main_speaker_text: What the main speaker said.
        speaker_name: Name of the interjecting speaker.

    Returns:
        Generated interjection text.
    """
    prompt = f"""You are {speaker_name}, listening to a podcast. The speaker just said:

"{main_speaker_text}"

Generate ONE SHORT natural interjection (2-6 words only) that shows you're engaged and listening.
Make it sound natural, not robotic. Examples: "Yeah, totally!", "That's wild!", "Wait, what?", "I see what you mean."

Respond with ONLY the interjection phrase, nothing else."""

    try:
        result_text = generate_text(
            [
                {"role": "system", "content": "You generate short conversational interjections."},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
        )
        if result_text:
            interjection = result_text.replace('"', "").replace("*", "").strip()
            logger.debug(f"LLM interjection: {interjection}")
            return interjection[:50]
    except Exception as e:
        logger.warning(f"LLM error {e}, using fallback")

    return get_fallback_interjection(main_speaker_text)


def get_fallback_interjection(main_speaker_text: str) -> str:
    """Fast fallback interjection pool if HF model is unavailable.

    Args:
        main_speaker_text: Context for selecting appropriate fallback.

    Returns:
        Fallback interjection string.
    """
    keywords = main_speaker_text.lower().split()

    if any(w in keywords for w in ["problem", "issue", "concerned", "worried"]):
        return random.choice(["That's concerning.", "I'm worried too.", "Yeah, I get it."])
    elif any(w in keywords for w in ["amazing", "incredible", "awesome", "great"]):
        return random.choice(["That's amazing!", "No way!", "That's insane!", "Right?"])
    elif any(w in keywords for w in ["think", "believe", "argue", "say"]):
        return random.choice(["I see your point.", "Fair argument.", "Makes sense."])
    else:
        return random.choice(["Yeah.", "Totally.", "Right.", "I hear you."])


def get_user_input():
    """Prompt user with file browser dialogs.

    Returns:
        Tuple of (json_path, output_dir, final_output_path, speaker_paths).
    """
    logger.info("=" * 60)
    logger.info("IndexTTS2 Multi-Speaker + LLM Interjections (On-Demand Calibration)")
    logger.info("=" * 60)

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    logger.info("Select JSON input file...")
    json_path = filedialog.askopenfilename(
        title="Select JSON input file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialdir=os.getcwd()
    )

    if not json_path:
        logger.error("No JSON file selected.")
        root.destroy()
        return None, None, None, None

    logger.info(f"Selected JSON file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    speaker_files_required = set()
    for entry in entries:
        speaker_files_required.add(entry["voice_sample"])

    logger.info("Required speaker WAV files from JSON:")
    for spk in sorted(speaker_files_required):
        logger.info(f"  - {spk}")

    speaker_paths = {}
    missing_files = []
    json_dir = os.path.dirname(json_path)

    # First pass: resolve absolute paths or paths relative to JSON file
    for spk_file in speaker_files_required:
        candidate = spk_file
        if os.path.isabs(candidate) and os.path.exists(candidate):
            speaker_paths[spk_file] = candidate
            logger.info(f"  Found (absolute): {spk_file}")
            continue

        json_relative = os.path.join(json_dir, spk_file)
        if os.path.exists(json_relative):
            speaker_paths[spk_file] = json_relative
            logger.info(f"  Found (json-relative): {spk_file}")
            continue

        missing_files.append(spk_file)

    # If some files are missing, prompt for a speaker directory
    if missing_files:
        logger.info("Select directory containing speaker WAV files...")
        speaker_dir = filedialog.askdirectory(
            title="Select directory containing speaker WAV files",
            initialdir=os.getcwd()
        )

        if not speaker_dir:
            logger.error("No speaker directory selected.")
            root.destroy()
            return None, None, None, None

        logger.debug(f"Selected speaker directory: {speaker_dir}")

        still_missing = []
        for spk_file in missing_files:
            full_path = os.path.join(speaker_dir, spk_file)
            if os.path.exists(full_path):
                speaker_paths[spk_file] = full_path
                logger.debug(f"  Found: {spk_file}")
            else:
                still_missing.append(spk_file)
                logger.warning(f"  Missing: {spk_file}")

        if still_missing:
            logger.error(f"{len(still_missing)} speaker WAV file(s) not found:")
            messagebox.showerror(
                "Missing Files",
                f"The following speaker files are missing:\n" + "\n".join(still_missing)
            )
            root.destroy()
            return None, None, None, None

    logger.info(f"All {len(speaker_files_required)} required speaker files found!")

    logger.info("Select output directory...")
    output_dir = filedialog.askdirectory(
        title="Select output directory",
        initialdir=os.getcwd()
    )

    if not output_dir:
        logger.error("No output directory selected.")
        root.destroy()
        return None, None, None, None

    logger.debug(f"Selected output directory: {output_dir}")

    root.destroy()

    while True:
        final_output_name = input("Enter final merged output filename (e.g., merged_podcast.wav): ").strip()
        if final_output_name:
            if not final_output_name.endswith(".wav"):
                final_output_name += ".wav"
            break
        logger.warning("Filename cannot be empty.")

    final_output_path = os.path.join(output_dir, final_output_name)

    return json_path, output_dir, final_output_path, speaker_paths


def synthesize_entry(entry: dict, idx: int, output_dir: str, speaker_paths: dict) -> str:
    """Generate TTS output for a single entry.

    Args:
        entry: JSON entry with text and speaker info.
        idx: Entry index.
        output_dir: Output directory.
        speaker_paths: Speaker filename to path mapping.

    Returns:
        Path to generated WAV file.
    """
    out_wav = os.path.join(output_dir, f"gen_{idx}.wav")
    logger.info(f"[{idx + 1}] Synthesizing: Speaker {entry['speaker']}")
    logger.debug(f"    Text: {entry['text'][:60]}...")

    spk_audio_path = speaker_paths[entry["voice_sample"]]

    tts.infer(
        spk_audio_prompt=spk_audio_path,
        text=entry["text"],
        output_path=out_wav,
        emo_alpha=entry.get("emo_alpha", 0.6),
        use_emo_text=entry.get("use_emo_text", True),
        emo_text=entry.get("emo_text", ""),
        use_random=False,
        verbose=False
    )
    return out_wav


def add_interjection_to_audio(main_audio_path: str, main_text: str, interjection_text: str,
                              interjector_voice_path: str, output_dir: str, idx: int,
                              trigger_info: dict) -> tuple:
    """Generate and overlay interjection on main speaker's audio.

    Args:
        main_audio_path: Path to main speaker audio.
        main_text: Main speaker's text.
        interjection_text: Text to synthesize as interjection.
        interjector_voice_path: Voice sample for interjection.
        output_dir: Output directory.
        idx: Segment index.
        trigger_info: Trigger timing information.

    Returns:
        Tuple of (mixed audio, metadata dict).
    """
    logger.info(f"   Adding interjection: '{interjection_text}'")

    interjection_wav = os.path.join(output_dir, f"interjection_{idx}.wav")
    tts.infer(
        spk_audio_prompt=interjector_voice_path,
        text=interjection_text,
        output_path=interjection_wav,
        emo_alpha=0.5,
        use_emo_text=True,
        emo_text="Casual listening agreement",
        use_random=False,
        verbose=False
    )

    main_audio = AudioSegment.from_wav(main_audio_path)
    interjection_audio = AudioSegment.from_wav(interjection_wav)

    # Smooth edges to prevent pops
    interjection_audio = interjection_audio.fade_in(20).fade_out(50)
    interjection_audio = interjection_audio - 2  # Slight volume reduction

    position_ms = trigger_info["position_ms"]
    mixed = main_audio.overlay(interjection_audio, position=position_ms)

    metadata = {
        "interjection_text": interjection_text,
        "trigger_word": trigger_info["trigger_word"],
        "trigger_category": trigger_info["trigger_category"],
        "trigger_position_ms": trigger_info["trigger_position_ms"],
        "trigger_position_percent": f"{trigger_info['trigger_position_percent']*100:.1f}%",
        "reaction_delay_ms": trigger_info["reaction_delay_ms"],
        "interjection_position_ms": position_ms,
        "interjection_position_percent": f"{(position_ms / trigger_info['audio_duration_ms'])*100:.1f}%",
        "audio_duration_ms": trigger_info["audio_duration_ms"]
    }

    logger.debug(f"      Trigger: '{trigger_info['trigger_word']}' ({trigger_info['trigger_category']}) @ {trigger_info['trigger_position_ms']}ms")
    logger.debug(f"      Position: {position_ms}ms")

    return mixed, metadata


def merge_with_interjections(json_entries: list, wav_files: list, output_dir: str,
                             final_output_path: str, speaker_paths: dict) -> list:
    """Main merge function with LLM-based interjections.

    Args:
        json_entries: List of JSON entries.
        wav_files: List of generated WAV paths.
        output_dir: Output directory.
        final_output_path: Final output file path.
        speaker_paths: Speaker paths mapping.

    Returns:
        List of interjection metadata.
    """
    logger.info("Merging with LLM interjections...")

    merged_audio = None
    interjection_count = 0
    interjections_log = []

    CROSSFADE_DURATION = 150

    for idx, (entry, wav_path) in enumerate(zip(json_entries, wav_files)):
        logger.info(f"Processing segment {idx + 1}/{len(wav_files)}...")

        main_audio = AudioSegment.from_wav(wav_path)

        if random.random() < 0.6 and idx > 0:
            other_speakers = {
                name: path for name, path in speaker_paths.items()
                if name != entry["voice_sample"]
            }

            if other_speakers:
                interjector_voice = random.choice(list(other_speakers.keys()))
                interjector_path = other_speakers[interjector_voice]
                interjector_idx = list(speaker_paths.keys()).index(interjector_voice)

                logger.debug(f"      Calibrating {entry['speaker']} for interjection detection...")
                seconds_per_word = calculate_seconds_per_word_from_audio(entry["text"], wav_path)

                trigger_info = calculate_interjection_timing_and_trigger(
                    entry["text"],
                    wav_path,
                    seconds_per_word
                )

                if trigger_info:
                    interjection_text = generate_interjection_llm(
                        entry["text"],
                        f"Speaker {interjector_idx}"
                    )

                    main_audio, interjection_metadata = add_interjection_to_audio(
                        wav_path,
                        entry["text"],
                        interjection_text,
                        interjector_path,
                        output_dir,
                        idx,
                        trigger_info
                    )

                    interjection_metadata["segment_idx"] = idx
                    interjection_metadata["main_speaker"] = entry["speaker"]
                    interjection_metadata["interjecting_speaker"] = interjector_idx
                    interjections_log.append(interjection_metadata)
                    interjection_count += 1

        if merged_audio is None:
            merged_audio = main_audio
        else:
            actual_crossfade = min(len(merged_audio), len(main_audio), CROSSFADE_DURATION)
            merged_audio = merged_audio.append(main_audio, crossfade=actual_crossfade)

    merged_audio.export(final_output_path, format="wav", bitrate="320k")
    logger.info(f"Merge complete with {interjection_count} interjections!")
    logger.info(f"Output: {final_output_path}")

    log_path = final_output_path.replace(".wav", "_interjections.json")
    with open(log_path, "w") as f:
        json.dump(interjections_log, f, indent=2)

    return interjections_log


def cleanup_temp_files(output_dir: str) -> None:
    """Remove temporary generated files.

    Args:
        output_dir: Directory containing temp files.
    """
    logger.info("Cleaning up temporary files...")
    for file in Path(output_dir).glob("gen_*.wav"):
        try:
            file.unlink()
            logger.debug(f"  Removed: {file.name}")
        except Exception:
            pass
    for file in Path(output_dir).glob("interjection_*.wav"):
        try:
            file.unlink()
        except Exception:
            pass


def main():
    """Main execution flow."""
    logger.info("=" * 60)
    logger.info("Checking HF Mistral 8B Setup...")
    logger.info("=" * 60)

    if not ensure_mistral_loaded():
        return

    json_path, output_dir, final_output_path, speaker_paths = get_user_input()

    if json_path is None:
        return

    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Input JSON: {json_path}")
    logger.info(f"  Output Directory: {output_dir}")
    logger.info(f"  Final Output: {final_output_path}")
    logger.info(f"  Interjection Model: {MODEL_ID} (HF 4-bit)")
    logger.info("=" * 60)

    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    logger.info(f"Processing {len(entries)} entries...")

    wav_files = []
    for idx, entry in enumerate(entries):
        try:
            wav_file = synthesize_entry(entry, idx, output_dir, speaker_paths)
            wav_files.append(wav_file)
        except Exception as e:
            logger.error(f"Error synthesizing entry {idx}: {e}")
            return

    try:
        interjections_log = merge_with_interjections(
            entries, wav_files, output_dir, final_output_path, speaker_paths
        )
    except Exception as e:
        logger.error(f"Error merging files: {e}")
        return

    cleanup_response = input("\nDelete temporary files? (y/n): ").strip().lower()
    if cleanup_response == 'y':
        cleanup_temp_files(output_dir)

    logger.info("=" * 60)
    logger.info("Synthesis complete!")
    logger.info(f"Final output: {final_output_path}")
    logger.info(f"Interjections recorded: {len(interjections_log)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
