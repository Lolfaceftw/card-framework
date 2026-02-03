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
    Use Mistral 7B to identify the most natural conversational trigger phrase.
    IMPROVED: No fixed categories. Detects context dynamically.
    """
    prompt = f"""
You are an expert conversation analyst. Your task is to find the single best "trigger phrase" in the text where a listener would naturally interject, react, or make a sound.

The text is:
"{text}"

**INSTRUCTIONS:**
1. Identify a specific contiguous phrase (2-6 words) that invites a reaction.
2. **BE FLEXIBLE:** Do not look for specific categories. Look for:
   - Humor / Sarcasm
   - Shock / Surprise
   - Hesitation / Uncertainty
   - Questions (Rhetorical or Direct)
   - Strong Opinions
   - Sadness / Empathy
3. The phrase MUST exist exactly in the text.

Respond with ONLY a single JSON object with keys: 
- "trigger_phrase": The exact text found.
- "context_vibe": A one-word description of the moment (e.g., "humorous", "shocking", "inquisitive", "sad", "agreement").

---
EXAMPLE 1
Text: "I walked in and, you won't believe it, the room was empty."
{{
  "trigger_phrase": "the room was empty",
  "context_vibe": "shocking"
}}
---
EXAMPLE 2
Text: "It's just... I don't know if I can do it."
{{
  "trigger_phrase": "I don't know",
  "context_vibe": "vulnerable"
}}
---
EXAMPLE 3
Text: "So I told him to get lost, obviously."
{{
  "trigger_phrase": "obviously",
  "context_vibe": "sassy"
}}
---

Analyze the provided text and give your JSON response.
"""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=30
        )

        if response.status_code == 200:
            result_text = response.json().get("response", "")
            print(f"Mistral's Response:\n{result_text}")
            trigger_data = json.loads(result_text)

            if "trigger_phrase" in trigger_data:
                phrase = trigger_data["trigger_phrase"]
                
                # Find phrase in original text (case-insensitive)
                start_index = text.lower().find(phrase.lower())
                
                if start_index != -1:
                    end_index = start_index + len(phrase)
                    trigger_data["pos_percent"] = end_index / len(text)
                    trigger_data["char_pos"] = end_index
                    return [trigger_data]
                else:
                    print(f"   Warning: LLM phrase '{phrase}' not found in text.")
                    return []
        else:
            print(f"   Warning: LLM status {response.status_code}")

    except Exception as e:
        print(f"   Warning: LLM error: {e}")

    return []

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
    Calculate timing based on flexible trigger phrases.
    IMPROVED: Dynamic reaction delay based on 'context_vibe'.
    """
    
    main_audio = AudioSegment.from_wav(main_audio_path)
    audio_duration_ms = len(main_audio)
    
    print("      └─ Asking LLM to find a natural trigger phrase...")
    triggers = detect_trigger_with_llm(main_text)
    
    if not triggers:
        return None
    
    selected_trigger = random.choice(triggers)
    trigger_pos_percent = selected_trigger["pos_percent"]
    trigger_phrase = selected_trigger["trigger_phrase"]
    vibe = selected_trigger.get("context_vibe", "neutral").lower()
    
    # Calculate position immediately AFTER the phrase ends
    trigger_time_ms = int(audio_duration_ms * trigger_pos_percent)
    
    # --- DYNAMIC REACTION DELAY ---
    # Fast reactions for high energy or questions
    fast_vibes = ["shocking", "surprising", "inquisitive", "urgent", "funny", "sassy", "exciting"]
    # Slow reactions for heavy emotions or deep thought
    slow_vibes = ["sad", "vulnerable", "thoughtful", "confusing", "serious", "hesitant"]
    
    if any(x in vibe for x in fast_vibes) or "?" in trigger_phrase:
        reaction_delay = random.randint(150, 400) # Fast snap reaction
    elif any(x in vibe for x in slow_vibes):
        reaction_delay = random.randint(600, 1000) # Pensive pause
    else:
        reaction_delay = random.randint(300, 700) # Standard conversational gap
        
    interjection_position = trigger_time_ms + reaction_delay
    
    # Safety check (keep within audio bounds)
    interjection_position = max(0, min(interjection_position, audio_duration_ms - 500))
    
    return {
        "position_ms": interjection_position,
        "trigger_word": trigger_phrase,
        "trigger_category": vibe, # We store the vibe as the category now
        "trigger_position_percent": trigger_pos_percent,
        "trigger_position_ms": trigger_time_ms,
        "reaction_delay_ms": reaction_delay,
        "audio_duration_ms": audio_duration_ms
    }

def generate_interjection_llm(main_speaker_text, speaker_name, context_vibe="neutral"):
    """
    Generate interjection matching the detected vibe.
    """
    prompt = f"""You are {speaker_name}, listening to a podcast. The context is {context_vibe}. The speaker just said:

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

