import json
import os
import random
import re
import time
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import numpy as np
import torch
from pydub import AudioSegment
from transformers import AutoModelForCausalLM, AutoTokenizer

from indextts.infer_v2 import IndexTTS2
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

# HF Mistral 8B (quantized 4-bit via bitsandbytes)
HF_MISTRAL_MODEL_ID = os.environ.get(
    "MISTRAL_MODEL_ID",
    "mistralai/Ministral-8B-Instruct-2410",
)
HF_MAX_NEW_TOKENS = int(os.environ.get("MISTRAL_MAX_NEW_TOKENS", "48"))

_HF_TOKENIZER = None
_HF_MODEL = None

# Speaker calibration cache: {(speaker_path, emo_text, emo_alpha): seconds_per_word}
SPEAKER_CALIBRATION = {}

def _ensure_hf_model():
    """Lazy-load the HF Mistral 8B model."""
    global _HF_MODEL, _HF_TOKENIZER  # noqa: PLW0603
    if _HF_MODEL is not None and _HF_TOKENIZER is not None:
        return True

    try:
        logger.info(f"Loading HF model: {HF_MISTRAL_MODEL_ID} (4-bit)")
        _HF_TOKENIZER = AutoTokenizer.from_pretrained(HF_MISTRAL_MODEL_ID)
        if _HF_TOKENIZER.pad_token is None:
            _HF_TOKENIZER.pad_token = _HF_TOKENIZER.eos_token

        _HF_MODEL = AutoModelForCausalLM.from_pretrained(
            HF_MISTRAL_MODEL_ID,
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        _HF_MODEL.eval()
        logger.info("✓ Mistral 8B loaded (HF 4-bit)")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to load HF Mistral 8B: {exc}")
        return False


def _build_prompt(messages):
    """Build a chat prompt string for Mistral."""
    if _HF_TOKENIZER is None:
        return ""
    try:
        return _HF_TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback to plain text concatenation
        joined = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            joined.append(f"{role.upper()}: {content}")
        joined.append("ASSISTANT:")
        return "\n".join(joined)


def _hf_generate(messages, max_new_tokens=32, temperature=0.7, top_p=0.9):
    """Generate text with the HF Mistral model."""
    if not _ensure_hf_model():
        return None

    prompt = _build_prompt(messages)
    if not prompt:
        return None

    inputs = _HF_TOKENIZER(prompt, return_tensors="pt")
    inputs = {k: v.to(_HF_MODEL.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _HF_MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=_HF_TOKENIZER.eos_token_id,
            eos_token_id=_HF_TOKENIZER.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return _HF_TOKENIZER.decode(generated, skip_special_tokens=True).strip()

def calibrate_speaker_speech_rate(speaker_voice_path, speaker_name, emo_alpha, emo_text, 
                                  use_emo_text, output_dir):
    """
    IMPROVED: Analyze actual speech rate of speaker by generating calibration audio
    WITH THE SAME EMOTIONAL PARAMETERS as the actual segment.
    
    This ensures accurate timing since emotion affects speech rate significantly.
    
    Returns: seconds_per_word (float)
    """
    
    # Create cache key: (speaker, emotion_profile)
    cache_key = (speaker_voice_path, emo_text, emo_alpha, use_emo_text)
    
    # Check if already calibrated with these exact parameters
    if cache_key in SPEAKER_CALIBRATION:
        logger.info(f"      └─ Using cached calibration: {speaker_name} (emo_alpha={emo_alpha}, emo_text='{emo_text[:30]}...')")
        return SPEAKER_CALIBRATION[cache_key]
    
    logger.info(f"      └─ Calibrating {speaker_name} with emotion: emo_alpha={emo_alpha}, emo_text='{emo_text[:30]}...'")
    
    # Calibration text: simple, known word count
    calibration_text = "The quick brown fox jumps over the lazy dog. This is a test."
    word_count = len(calibration_text.split())  # 12 words
    
    calibration_wav = os.path.join(output_dir, f"calibration_{speaker_name}_{hash(cache_key) % 10000}.wav")
    
    # Generate calibration audio WITH SAME EMOTIONAL PARAMETERS as the actual segment
    tts.infer(
        spk_audio_prompt=speaker_voice_path,
        text=calibration_text,
        output_path=calibration_wav,
        emo_alpha=emo_alpha,  # Use actual emo_alpha from JSON
        use_emo_text=use_emo_text,  # Use actual setting
        emo_text=emo_text if use_emo_text else "",  # Use actual emotion description
        use_random=False,
        verbose=False
    )
    
    # Measure actual duration
    calibration_audio = AudioSegment.from_wav(calibration_wav)
    audio_duration_seconds = len(calibration_audio) / 1000.0  # Convert ms to seconds
    
    # Calculate actual seconds per word for this speaker with this emotion
    seconds_per_word = audio_duration_seconds / word_count
    
    # Cache result
    SPEAKER_CALIBRATION[cache_key] = seconds_per_word
    
    logger.info(f"         └─ Calibrated: {seconds_per_word:.4f} sec/word")
    logger.info(f"            (Audio: {audio_duration_seconds:.2f}s for {word_count} words with emotion: {emo_text})")
    
    # Clean up calibration file
    try:
        os.remove(calibration_wav)
    except:
        pass
    
    return seconds_per_word

def detect_trigger_words(text):
    """
    Detect trigger words/phrases that warrant an interjection.
    Returns list of trigger words and their approximate positions in text.
    """
    trigger_patterns = {
        "question": [r"\?", r"is ", r"what ", r"why ", r"how "],
        "statement": [r"amazing", r"incredible", r"terrible", r"shocking", r"unbelievable"],
        "problem": [r"issue", r"problem", r"concerning", r"worried", r"concerned"],
        "agreement_seek": [r"don't you", r"isn't it", r"right\?", r"yeah\?"],
    }
    
    text_lower = text.lower()
    detected_triggers = []
    
    for category, patterns in trigger_patterns.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                pos_percent = match.start() / len(text)
                detected_triggers.append({
                    "category": category,
                    "word": match.group(),
                    "pos_percent": pos_percent,
                    "char_pos": match.start()
                })
    
    return detected_triggers


def _extract_first_json(text):
    """Extract first JSON object from text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def detect_trigger_llm(text):
    """Detect trigger word using Mistral 8B. Returns a single trigger dict or None."""
    system_msg = {
        "role": "system",
        "content": (
            "You are a conversational supervisor. Identify the best trigger word/phrase for a listener interjection. "
            "Return STRICT JSON only."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            "Analyze the text and return JSON ONLY:\n"
            "{ \"trigger_word\": \"word\", \"char_pos\": 12, \"category\": \"statement\" }\n\n"
            f"Text: {text}"
        ),
    }
    raw = _hf_generate([system_msg, user_msg], max_new_tokens=HF_MAX_NEW_TOKENS, temperature=0.2, top_p=0.9)
    if not raw:
        return None

    payload = _extract_first_json(raw)
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    trigger_word = str(data.get("trigger_word", "")).strip()
    char_pos = data.get("char_pos")
    category = str(data.get("category", "statement")).strip() or "statement"

    if not trigger_word or not isinstance(char_pos, int):
        return None

    pos_percent = char_pos / max(1, len(text))
    return {
        "category": category,
        "word": trigger_word,
        "pos_percent": pos_percent,
        "char_pos": char_pos,
    }

def estimate_word_timing(text, audio_duration_ms, seconds_per_word):
    """
    Estimate timing of words in synthesized audio using speaker's emotion-specific speech rate.
    
    IMPROVED: Uses calibrated seconds_per_word (with emotion) for accurate timing.
    """
    words = text.split()
    word_timings = {}
    
    # Use emotion-calibrated speech rate
    current_time = 0
    for idx, word in enumerate(words):
        word_duration = seconds_per_word * 1000  # Convert to ms
        word_timings[word.lower()] = {
            "start_ms": int(current_time),
            "end_ms": int(current_time + word_duration),
            "position": idx / len(words)
        }
        current_time += word_duration
    
    return word_timings

def calculate_interjection_timing(main_text, main_audio_path, seconds_per_word):
    """
    Calculate optimal interjection timing based on trigger words and emotion-calibrated speech rate.
    
    Returns millisecond position where interjection should occur.
    """
    
    main_audio = AudioSegment.from_wav(main_audio_path)
    audio_duration_ms = len(main_audio)
    
    # Detect trigger words in the main speaker's text
    triggers = []
    llm_trigger = detect_trigger_llm(main_text)
    if llm_trigger is not None:
        triggers = [llm_trigger]
    else:
        triggers = detect_trigger_words(main_text)
    
    if not triggers:
        # Fallback: use mid-speech if no triggers detected
        return int(audio_duration_ms * 0.5) + random.randint(-500, 500)
    
    # Select a random trigger
    selected_trigger = random.choice(triggers)
    trigger_pos_percent = selected_trigger["pos_percent"]
    
    # Estimate word timings using emotion-calibrated speech rate
    word_timings = estimate_word_timing(main_text, audio_duration_ms, seconds_per_word)
    
    # Calculate position AFTER the trigger word
    trigger_time_ms = int(audio_duration_ms * trigger_pos_percent)
    reaction_delay = random.randint(300, 800)
    interjection_position = trigger_time_ms + reaction_delay
    
    # Safety check: ensure position is within audio bounds
    interjection_position = max(0, min(interjection_position, audio_duration_ms - 500))
    
    logger.info(f"      └─ Trigger detected: '{selected_trigger['word']}' ({selected_trigger['category']}) @ {trigger_time_ms}ms")
    logger.info(f"      └─ Interjection scheduled at: {interjection_position}ms (after {reaction_delay}ms reaction)")
    
    return interjection_position

def generate_interjection_llm(main_speaker_text, speaker_name):
    """
    Generate interjection using Mistral 8B (HF 4-bit).
    Uses speaker context to personalize the response.
    """
    
    prompt = f"""You are {speaker_name}, listening to a podcast. The speaker just said:

"{main_speaker_text[:200]}"

Generate ONE SHORT natural interjection (2-6 words only) that shows you're engaged and listening.
Make it sound natural, not robotic. Examples: "Yeah, totally!", "That's wild!", "Wait, what?", "I see what you mean."

Respond with ONLY the interjection phrase, nothing else."""
    
    try:
        result = _hf_generate(
            [
                {"role": "system", "content": "You generate short conversational interjections."},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
        )
        if result:
            interjection = result.replace('"', "").replace("*", "").strip()
            return interjection[:50]
    except Exception as e:
        logger.info(f"   Warning: LLM error {e}, using fallback")
    
    return get_fallback_interjection(main_speaker_text)

def get_fallback_interjection(main_speaker_text):
    """Fast fallback interjection pool if HF model is unavailable."""
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
    """Prompt user with file browser dialogs"""
    logger.info("=" * 60)
    logger.info("IndexTTS2 Multi-Speaker + LLM Interjections (Emotion-Aware Calibration)")
    logger.info("=" * 60)
    
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    logger.info("\nSelect JSON input file...")
    json_path = filedialog.askopenfilename(
        title="Select JSON input file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialdir=os.getcwd()
    )
    
    if not json_path:
        logger.info("Error: No JSON file selected.")
        root.destroy()
        return None, None, None, None
    
    logger.info(f"Selected JSON file: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    speaker_files_required = set(entry["voice_sample"] for entry in entries)

    logger.info("\nRequired speaker WAV files from JSON:")
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
            logger.info(f"  ✓ Found (absolute): {spk_file}")
            continue

        json_relative = os.path.join(json_dir, spk_file)
        if os.path.exists(json_relative):
            speaker_paths[spk_file] = json_relative
            logger.info(f"  ✓ Found (json-relative): {spk_file}")
            continue

        missing_files.append(spk_file)

    # If some files are missing, prompt for a speaker directory
    if missing_files:
        logger.info("\nSelect directory containing speaker WAV files...")
        speaker_dir = filedialog.askdirectory(
            title="Select directory containing speaker WAV files",
            initialdir=os.getcwd()
        )

        if not speaker_dir:
            logger.info("Error: No speaker directory selected.")
            root.destroy()
            return None, None, None, None

        logger.info(f"Selected speaker directory: {speaker_dir}")

        still_missing = []
        for spk_file in missing_files:
            full_path = os.path.join(speaker_dir, spk_file)
            if os.path.exists(full_path):
                speaker_paths[spk_file] = full_path
                logger.info(f"  ✓ Found: {spk_file}")
            else:
                still_missing.append(spk_file)
                logger.info(f"  ✗ Missing: {spk_file}")

        if still_missing:
            logger.info(f"\nError: {len(still_missing)} speaker WAV file(s) not found:")
            messagebox.showerror(
                "Missing Files",
                "The following speaker files are missing:\n" + "\n".join(still_missing),
            )
            root.destroy()
            return None, None, None, None

    logger.info(f"\n✓ All {len(speaker_files_required)} required speaker files found!")
    
    logger.info("\nSelect output directory...")
    output_dir = filedialog.askdirectory(
        title="Select output directory",
        initialdir=os.getcwd()
    )
    
    if not output_dir:
        logger.info("Error: No output directory selected.")
        root.destroy()
        return None, None, None, None
    
    logger.info(f"Selected output directory: {output_dir}")
    
    root.destroy()
    
    while True:
        final_output_name = input("Enter final merged output filename (e.g., merged_podcast.wav): ").strip()
        if final_output_name:
            if not final_output_name.endswith(".wav"):
                final_output_name += ".wav"
            break
        logger.info("Error: Filename cannot be empty.")
    
    final_output_path = os.path.join(output_dir, final_output_name)
    
    return json_path, output_dir, final_output_path, speaker_paths

def synthesize_entry(entry, idx, output_dir, speaker_paths):
    """Generate TTS output for a single entry"""
    out_wav = os.path.join(output_dir, f"gen_{idx}.wav")
    logger.info(f"\n[{idx + 1}] Synthesizing: Speaker {entry['speaker']}")
    logger.info(f"    Text: {entry['text'][:60]}...")
    
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

def add_interjection_to_audio(main_audio_path, main_text, interjection_text, interjector_voice_path, 
                              output_dir, idx, seconds_per_word):
    """
    Generate and overlay interjection on main speaker's audio.
    Uses emotion-calibrated timing.
    """
    logger.info(f"   └─ Adding LLM-generated interjection: '{interjection_text}'")
    
    # Generate interjection audio
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
    
    # Load both audios (no normalization)
    main_audio = AudioSegment.from_wav(main_audio_path)
    interjection_audio = AudioSegment.from_wav(interjection_wav)
    
    # Calculate timing based on trigger words with emotion-calibrated speech rate
    position_ms = calculate_interjection_timing(main_text, main_audio_path, seconds_per_word)
    
    # Overlay at calculated position (no attenuation)
    mixed = main_audio.overlay(interjection_audio, position=position_ms)
    
    return mixed

def merge_with_interjections(json_entries, wav_files, output_dir, final_output_path, speaker_paths, speaker_calibration):
    """
    Main merge function with LLM-based interjections using emotion-aware calibrated speech rates.
    """
    
    logger.info("\nMerging with LLM interjections (emotion-calibrated timing)...")
    
    merged_audio = None
    interjection_count = 0
    
    for idx, (entry, wav_path) in enumerate(zip(json_entries, wav_files)):
        logger.info(f"\n  Processing segment {idx + 1}/{len(wav_files)}...")
        
        # Load main speaker audio (no normalization)
        main_audio = AudioSegment.from_wav(wav_path)
        
        # Get emotion-calibrated speech rate for this specific segment
        main_speaker_path = speaker_paths[entry["voice_sample"]]
        emo_alpha = entry.get("emo_alpha", 0.6)
        use_emo_text = entry.get("use_emo_text", True)
        emo_text = entry.get("emo_text", "")
        
        cache_key = (main_speaker_path, emo_text, emo_alpha, use_emo_text)
        main_speaker_rate = speaker_calibration.get(cache_key, 0.1)
        
        # Decide if interjection should happen
        if random.random() < 0.6 and idx > 0:
            other_speakers = {
                name: path for name, path in speaker_paths.items() 
                if name != entry["voice_sample"]
            }
            
            if other_speakers:
                interjector_voice = random.choice(list(other_speakers.keys()))
                interjector_path = other_speakers[interjector_voice]
                interjector_idx = list(speaker_paths.keys()).index(interjector_voice)
                
                # Generate interjection using LLM
                interjection_text = generate_interjection_llm(
                    entry["text"],
                    f"Speaker {interjector_idx}"
                )
                
                # Add overlapping interjection with emotion-calibrated timing
                main_audio = add_interjection_to_audio(
                    wav_path, 
                    entry["text"],
                    interjection_text,
                    interjector_path,
                    output_dir,
                    idx,
                    main_speaker_rate
                )
                interjection_count += 1
        
        # Merge with previous audio
        if merged_audio is None:
            merged_audio = main_audio
        else:
            pause = AudioSegment.silent(duration=400)
            merged_audio = merged_audio.append(pause, crossfade=0)
            merged_audio = merged_audio.append(main_audio, crossfade=300)
    
    # Export
    merged_audio.export(final_output_path, format="wav", bitrate="320k")
    logger.info(f"\n✓ Merge complete with {interjection_count} LLM interjections (emotion-aware timing)!")
    logger.info(f"✓ Output: {final_output_path}")

def cleanup_temp_files(output_dir):
    """Remove temporary generated files"""
    logger.info("\nCleaning up temporary files...")
    for file in Path(output_dir).glob("gen_*.wav"):
        try:
            file.unlink()
            logger.info(f"  Removed: {file.name}")
        except:
            pass
    for file in Path(output_dir).glob("interjection_*.wav"):
        try:
            file.unlink()
        except:
            pass
    for file in Path(output_dir).glob("calibration_*.wav"):
        try:
            file.unlink()
        except:
            pass

def main():
    """Main execution flow"""
    
    logger.info("\n" + "="*60)
    logger.info("Checking HF Mistral 8B Setup...")
    logger.info("="*60)
    
    if not _ensure_hf_model():
        return
    
    json_path, output_dir, final_output_path, speaker_paths = get_user_input()
    
    if json_path is None:
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Configuration:")
    logger.info(f"  Input JSON: {json_path}")
    logger.info(f"  Output Directory: {output_dir}")
    logger.info(f"  Final Output: {final_output_path}")
    logger.info(f"  Interjection Model: {HF_MISTRAL_MODEL_ID} (HF 4-bit)")
    logger.info(f"  Timing Strategy: Smart trigger word + emotion-calibrated speech rate")
    logger.info(f"  Audio Levels: Preserved (no normalization)")
    logger.info(f"  Interjection Attenuation: NONE (full volume)")
    logger.info(f"{'='*60}\n")
    
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    logger.info(f"Processing {len(entries)} entries with emotion-aware interjections...\n")
    
    # CALIBRATION PHASE: Measure each speaker's speech rate per emotion
    logger.info("CALIBRATION PHASE: Measuring each speaker's speech rate for each emotion profile...\n")
    speaker_calibration = {}
    unique_emotion_profiles = {}
    
    # Collect unique emotion profiles
    for entry in entries:
        speaker = entry["voice_sample"]
        emo_alpha = entry.get("emo_alpha", 0.6)
        use_emo_text = entry.get("use_emo_text", True)
        emo_text = entry.get("emo_text", "")
        
        cache_key = (speaker, emo_text, emo_alpha, use_emo_text)
        if cache_key not in unique_emotion_profiles:
            unique_emotion_profiles[cache_key] = {
                "speaker": speaker,
                "emo_alpha": emo_alpha,
                "use_emo_text": use_emo_text,
                "emo_text": emo_text
            }
    
    # Calibrate each unique emotion profile
    for cache_key, profile in unique_emotion_profiles.items():
        speaker_name = profile["speaker"].replace(".wav", "")
        speaker_path = speaker_paths[profile["speaker"]]
        
        # Calibrate this specific emotion profile
        secs_per_word = calibrate_speaker_speech_rate(
            speaker_path,
            speaker_name,
            profile["emo_alpha"],
            profile["emo_text"],
            profile["use_emo_text"],
            output_dir
        )
        speaker_calibration[cache_key] = secs_per_word
    
    logger.info(f"\n✓ Calibration complete for {len(speaker_calibration)} emotion profile(s)\n")
    
    # SYNTHESIS PHASE: Generate all audio segments
    wav_files = []
    for idx, entry in enumerate(entries):
        try:
            wav_file = synthesize_entry(entry, idx, output_dir, speaker_paths)
            wav_files.append(wav_file)
        except Exception as e:
            logger.error(f"Error synthesizing entry {idx}: {e}")
            return
    
    # MERGE PHASE: Add interjections and merge
    try:
        merge_with_interjections(entries, wav_files, output_dir, final_output_path, speaker_paths, speaker_calibration)
    except Exception as e:
        logger.error(f"Error merging files: {e}")
        return
    
    cleanup_response = input("\nDelete temporary files? (y/n): ").strip().lower()
    if cleanup_response == 'y':
        cleanup_temp_files(output_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ Synthesis complete!")
    logger.info(f"Final output: {final_output_path}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()

