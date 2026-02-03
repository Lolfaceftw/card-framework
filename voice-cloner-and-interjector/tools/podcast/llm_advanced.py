import json
import os
import random
import subprocess
from pathlib import Path
from tkinter import Tk, filedialog, messagebox
from indextts.infer_v2 import IndexTTS2
from pydub import AudioSegment
import requests


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

def detect_trigger_with_llm(text):
    """
    Use Mistral 7B to identify the best trigger point for an interjection.
    Returns a dictionary with trigger info or None if it fails.
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
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json" # Ollama has a dedicated JSON mode
            },
            timeout=30
        )

        if response.status_code == 200:
            result_text = response.json().get("response", "")
            print(f"Mistral's Response:\n{result_text}")
            # The 'format: json' mode in Ollama returns a clean JSON string
            trigger_data = json.loads(result_text)

            # Basic validation
            if "trigger_word" in trigger_data and "char_pos" in trigger_data:
                # Calculate the position percentage required by the downstream function
                trigger_data["pos_percent"] = trigger_data["char_pos"] / len(text)
                return [trigger_data] # Return as a list to match the old function's format
        else:
            print(f"   Warning: LLM trigger detection failed with status {response.status_code}")

    except Exception as e:
        print(f"   Warning: LLM trigger detection error: {e}")

    return [] # Return an empty list on failure, so the script can handle it gracefully

def calculate_seconds_per_word_from_audio(text, audio_path):
    """
    Calculate actual seconds-per-word directly from synthesized audio.
    IMPROVED: Uses real audio measurement, not estimated.
    
    Returns: seconds_per_word (float)
    """
    audio = AudioSegment.from_wav(audio_path)
    audio_duration_seconds = len(audio) / 1000.0
    word_count = len(text.split())
    seconds_per_word = audio_duration_seconds / word_count
    
    return seconds_per_word

def calculate_interjection_timing_and_trigger(main_text, main_audio_path, seconds_per_word):
    """
    Calculate optimal interjection timing based on trigger words.
    IMPROVED: Returns trigger word info along with timing.
    
    Returns: {
        "position_ms": int,
        "trigger_word": str,
        "trigger_category": str,
        "trigger_position_percent": float,
        "reaction_delay_ms": int
    }
    """
    
    main_audio = AudioSegment.from_wav(main_audio_path)
    audio_duration_ms = len(main_audio)
    
    # Detect trigger words in the main speaker's text
    print("      └─ Asking LLM to find a trigger point...")
    triggers = detect_trigger_with_llm(main_text) # <-- Changed to the new function
    
    if not triggers:
        # Fallback: use mid-speech if no triggers detected
        return None
    
    # Select a random trigger
    selected_trigger = random.choice(triggers)
    trigger_pos_percent = selected_trigger["pos_percent"]
    
    # Calculate position AFTER the trigger word
    trigger_time_ms = int(audio_duration_ms * trigger_pos_percent)
    reaction_delay = random.randint(300, 800)
    interjection_position = trigger_time_ms + reaction_delay
    
    # Safety check: ensure position is within audio bounds
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

def generate_interjection_llm(main_speaker_text, speaker_name):
    """
    Generate interjection using Mistral 7B via Ollama.
    """
    
    prompt = f"""You are {speaker_name}, listening to a podcast. The speaker just said:

"{main_speaker_text}"

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
    print("IndexTTS2 Multi-Speaker + LLM Interjections (On-Demand Calibration)")
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
                              output_dir, idx, trigger_info):
    """
    Generate and overlay interjection on main speaker's audio.
    IMPROVED: Adds fade-in/out to prevent clicks and slight volume adjustment.
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
    
    # Load audios
    main_audio = AudioSegment.from_wav(main_audio_path)
    interjection_audio = AudioSegment.from_wav(interjection_wav)
    
    # --- CROSS-FADE / SMOOTHING LOGIC ---
    # 1. Smooth the edges of the interjection to prevent "pops"
    #    (20ms fade in, 50ms fade out)
    interjection_audio = interjection_audio.fade_in(20).fade_out(50)
    
    # 2. (Optional) Lower interjection volume slightly (-2dB) so it doesn't overpower
    interjection_audio = interjection_audio - 2
    # ------------------------------------

    # Get position from trigger info
    position_ms = trigger_info["position_ms"]
    
    # Overlay at calculated position
    mixed = main_audio.overlay(interjection_audio, position=position_ms)
    
    # Create metadata
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
    
    print(f"      └─ Trigger word: '{trigger_info['trigger_word']}' ({trigger_info['trigger_category']}) @ {trigger_info['trigger_position_ms']}ms")
    print(f"      └─ Interjection position: {position_ms}ms")
    
    return mixed, metadata

def merge_with_interjections(json_entries, wav_files, output_dir, final_output_path, speaker_paths):
    """
    Main merge function with LLM-based interjections.
    IMPROVED: Implements true cross-fading between main segments.
    """
    
    print("\nMerging with LLM interjections (on-demand calibration)...")
    
    merged_audio = None
    interjection_count = 0
    interjections_log = []
    
    # Configuration for segment transitions
    CROSSFADE_DURATION = 150  # ms of overlap between speakers
    
    for idx, (entry, wav_path) in enumerate(zip(json_entries, wav_files)):
        print(f"\n  Processing segment {idx + 1}/{len(wav_files)}...")
        
        # Load main speaker audio
        main_audio = AudioSegment.from_wav(wav_path)
        
        # --- INTERJECTION LOGIC (Kept same, just calling the updated function) ---
        if random.random() < 0.6 and idx > 0:
            other_speakers = {
                name: path for name, path in speaker_paths.items() 
                if name != entry["voice_sample"]
            }
            
            if other_speakers:
                interjector_voice = random.choice(list(other_speakers.keys()))
                interjector_path = other_speakers[interjector_voice]
                interjector_idx = list(speaker_paths.keys()).index(interjector_voice)
                
                print(f"      └─ Calibrating {entry['speaker']} for interjection detection...")
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
                    
                    # This calls our updated function with smoothing
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
        
        # --- MERGE LOGIC (UPDATED) ---
        if merged_audio is None:
            merged_audio = main_audio
        else:
            # Pydub's append with crossfade automatically overlaps the end of A 
            # and start of B by X milliseconds.
            
            # Safety check: Crossfade cannot be longer than the audio clips
            actual_crossfade = min(len(merged_audio), len(main_audio), CROSSFADE_DURATION)
            
            # If you want a tiny pause BEFORE the crossfade starts (optional):
            # silence_padding = AudioSegment.silent(duration=100) 
            # merged_audio = merged_audio + silence_padding
            
            merged_audio = merged_audio.append(main_audio, crossfade=actual_crossfade)
            
    # Export
    merged_audio.export(final_output_path, format="wav", bitrate="320k")
    print(f"\n✓ Merge complete with {interjection_count} LLM interjections!")
    print(f"✓ Output: {final_output_path}")
    
    # Save interjections log
    log_path = final_output_path.replace(".wav", "_interjections.json")
    with open(log_path, "w") as f:
        json.dump(interjections_log, f, indent=2)
    
    return interjections_log

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
    print(f"  Calibration: On-demand (only for segments with interjections)")
    print(f"  Timing: Actual audio measurement + trigger word detection")
    print(f"  Audio Levels: Preserved (no normalization)")
    print(f"  Interjection Attenuation: NONE (full volume)")
    print(f"{'='*60}\n")
    
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    print(f"Processing {len(entries)} entries with on-demand interjection calibration...\n")
    
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
        interjections_log = merge_with_interjections(entries, wav_files, output_dir, final_output_path, speaker_paths)
    except Exception as e:
        print(f"Error merging files: {e}")
        return
    
    cleanup_response = input("\nDelete temporary files? (y/n): ").strip().lower()
    if cleanup_response == 'y':
        cleanup_temp_files(output_dir)
    
    print(f"\n{'='*60}")
    print("✓ Synthesis complete!")
    print(f"Final output: {final_output_path}")
    print(f"Interjections recorded: {len(interjections_log)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

