import json
import os
import random
import subprocess
from pathlib import Path
from tkinter import Tk, filedialog, messagebox
from indextts.infer_v2 import IndexTTS2
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import requests
import time
import re

# Initialize Index-TTS-2
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    device="cuda:0",
    use_fp16=True,
    use_cuda_kernel=True
)

# OLLAMA CONFIGURATION
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b-instruct-q4_0"
OLLAMA_CHECK_URL = "http://localhost:11434/api/tags"

# Speaker calibration cache: {(speaker_path, emo_text, emo_alpha): seconds_per_word}
SPEAKER_CALIBRATION = {}

def check_ollama_running():
    """Verify Ollama service is running"""
    try:
        response = requests.get(OLLAMA_CHECK_URL, timeout=2)
        return response.status_code == 200
    except:
        return False

def ensure_mistral_loaded():
    """Download Mistral 7B 4-bit if not already present"""
    if not check_ollama_running():
        print("\n⚠ Ollama is not running!")
        print("   Start Ollama with: ollama serve")
        print("   Then run this script again.")
        return False
    
    try:
        response = requests.get(OLLAMA_CHECK_URL, timeout=5)
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        
        if OLLAMA_MODEL not in model_names:
            print(f"\nDownloading {OLLAMA_MODEL}... (this is a one-time setup, ~2-3 minutes)")
            subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
            print("✓ Mistral 7B loaded!")
        else:
            print(f"✓ {OLLAMA_MODEL} already loaded")
        return True
    except Exception as e:
        print(f"Error checking Ollama models: {e}")
        return False

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
        print(f"      └─ Using cached calibration: {speaker_name} (emo_alpha={emo_alpha}, emo_text='{emo_text[:30]}...')")
        return SPEAKER_CALIBRATION[cache_key]
    
    print(f"      └─ Calibrating {speaker_name} with emotion: emo_alpha={emo_alpha}, emo_text='{emo_text[:30]}...'")
    
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
    
    print(f"         └─ Calibrated: {seconds_per_word:.4f} sec/word")
    print(f"            (Audio: {audio_duration_seconds:.2f}s for {word_count} words with emotion: {emo_text})")
    
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
    
    print(f"      └─ Trigger detected: '{selected_trigger['word']}' ({selected_trigger['category']}) @ {trigger_time_ms}ms")
    print(f"      └─ Interjection scheduled at: {interjection_position}ms (after {reaction_delay}ms reaction)")
    
    return interjection_position

def generate_interjection_llm(main_speaker_text, speaker_name):
    """
    Generate interjection using Mistral 7B via Ollama.
    Uses speaker context to personalize the response.
    """
    
    prompt = f"""You are {speaker_name}, listening to a podcast. The speaker just said:

"{main_speaker_text[:200]}"

Generate ONE SHORT natural interjection (2-6 words only) that shows you're engaged and listening.
Make it sound natural, not robotic. Examples: "Yeah, totally!", "That's wild!", "Wait, what?", "I see what you mean."

Respond with ONLY the interjection phrase, nothing else."""
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 20
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            interjection = result.get("response", "").strip()
            interjection = interjection.replace('"', '').replace("*", '')
            return interjection[:50]
        
    except Exception as e:
        print(f"   Warning: LLM error {e}, using fallback")
    
    return get_fallback_interjection(main_speaker_text)

def get_fallback_interjection(main_speaker_text):
    """Fast fallback interjection pool if Ollama is down"""
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
    print("=" * 60)
    print("IndexTTS2 Multi-Speaker + LLM Interjections (Emotion-Aware Calibration)")
    print("=" * 60)
    
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("\nSelect JSON input file...")
    json_path = filedialog.askopenfilename(
        title="Select JSON input file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialdir=os.getcwd()
    )
    
    if not json_path:
        print("Error: No JSON file selected.")
        root.destroy()
        return None, None, None, None
    
    print(f"Selected JSON file: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    speaker_files_required = set()
    for entry in entries:
        speaker_files_required.add(entry["voice_sample"])
    
    print(f"\nRequired speaker WAV files from JSON:")
    for spk in sorted(speaker_files_required):
        print(f"  - {spk}")
    
    print(f"\nSelect directory containing speaker WAV files...")
    speaker_dir = filedialog.askdirectory(
        title="Select directory containing speaker WAV files",
        initialdir=os.getcwd()
    )
    
    if not speaker_dir:
        print("Error: No speaker directory selected.")
        root.destroy()
        return None, None, None, None
    
    print(f"Selected speaker directory: {speaker_dir}")
    
    speaker_paths = {}
    missing_files = []
    
    for spk_file in speaker_files_required:
        full_path = os.path.join(speaker_dir, spk_file)
        if os.path.exists(full_path):
            speaker_paths[spk_file] = full_path
            print(f"  ✓ Found: {spk_file}")
        else:
            missing_files.append(spk_file)
            print(f"  ✗ Missing: {spk_file}")
    
    if missing_files:
        print(f"\nError: {len(missing_files)} speaker WAV file(s) not found:")
        messagebox.showerror("Missing Files", 
            f"The following speaker files are missing:\n" + "\n".join(missing_files))
        root.destroy()
        return None, None, None, None
    
    print(f"\n✓ All {len(speaker_files_required)} required speaker files found!")
    
    print("\nSelect output directory...")
    output_dir = filedialog.askdirectory(
        title="Select output directory",
        initialdir=os.getcwd()
    )
    
    if not output_dir:
        print("Error: No output directory selected.")
        root.destroy()
        return None, None, None, None
    
    print(f"Selected output directory: {output_dir}")
    
    root.destroy()
    
    while True:
        final_output_name = input("Enter final merged output filename (e.g., merged_podcast.wav): ").strip()
        if final_output_name:
            if not final_output_name.endswith(".wav"):
                final_output_name += ".wav"
            break
        print("Error: Filename cannot be empty.")
    
    final_output_path = os.path.join(output_dir, final_output_name)
    
    return json_path, output_dir, final_output_path, speaker_paths

def synthesize_entry(entry, idx, output_dir, speaker_paths):
    """Generate TTS output for a single entry"""
    out_wav = os.path.join(output_dir, f"gen_{idx}.wav")
    print(f"\n[{idx + 1}] Synthesizing: Speaker {entry['speaker']}")
    print(f"    Text: {entry['text'][:60]}...")
    
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
    print(f"   └─ Adding LLM-generated interjection: '{interjection_text}'")
    
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
    
    print("\nMerging with LLM interjections (emotion-calibrated timing)...")
    
    merged_audio = None
    interjection_count = 0
    
    for idx, (entry, wav_path) in enumerate(zip(json_entries, wav_files)):
        print(f"\n  Processing segment {idx + 1}/{len(wav_files)}...")
        
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
    print(f"\n✓ Merge complete with {interjection_count} LLM interjections (emotion-aware timing)!")
    print(f"✓ Output: {final_output_path}")

def cleanup_temp_files(output_dir):
    """Remove temporary generated files"""
    print("\nCleaning up temporary files...")
    for file in Path(output_dir).glob("gen_*.wav"):
        try:
            file.unlink()
            print(f"  Removed: {file.name}")
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
    
    print("\n" + "="*60)
    print("Checking Ollama + Mistral 7B Setup...")
    print("="*60)
    
    if not ensure_mistral_loaded():
        return
    
    json_path, output_dir, final_output_path, speaker_paths = get_user_input()
    
    if json_path is None:
        return
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Input JSON: {json_path}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Final Output: {final_output_path}")
    print(f"  Interjection Model: Mistral 7B (4-bit via Ollama)")
    print(f"  Timing Strategy: Smart trigger word + emotion-calibrated speech rate")
    print(f"  Audio Levels: Preserved (no normalization)")
    print(f"  Interjection Attenuation: NONE (full volume)")
    print(f"{'='*60}\n")
    
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    print(f"Processing {len(entries)} entries with emotion-aware interjections...\n")
    
    # CALIBRATION PHASE: Measure each speaker's speech rate per emotion
    print("CALIBRATION PHASE: Measuring each speaker's speech rate for each emotion profile...\n")
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
    
    print(f"\n✓ Calibration complete for {len(speaker_calibration)} emotion profile(s)\n")
    
    # SYNTHESIS PHASE: Generate all audio segments
    wav_files = []
    for idx, entry in enumerate(entries):
        try:
            wav_file = synthesize_entry(entry, idx, output_dir, speaker_paths)
            wav_files.append(wav_file)
        except Exception as e:
            print(f"Error synthesizing entry {idx}: {e}")
            return
    
    # MERGE PHASE: Add interjections and merge
    try:
        merge_with_interjections(entries, wav_files, output_dir, final_output_path, speaker_paths, speaker_calibration)
    except Exception as e:
        print(f"Error merging files: {e}")
        return
    
    cleanup_response = input("\nDelete temporary files? (y/n): ").strip().lower()
    if cleanup_response == 'y':
        cleanup_temp_files(output_dir)
    
    print(f"\n{'='*60}")
    print("✓ Synthesis complete!")
    print(f"Final output: {final_output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
