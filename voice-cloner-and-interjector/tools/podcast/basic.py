import json
import os
from pathlib import Path
from tkinter import Tk, filedialog, messagebox
from indextts.infer_v2 import IndexTTS2
import soundfile as sf
import numpy as np

# Load configuration for IndexTTS2
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    device="cuda:0",
    use_fp16=True,
    use_cuda_kernel=True
)

def get_user_input():
    """Prompt user with file browser dialogs for JSON input, output directory, and speaker WAV files"""
    print("=" * 60)
    print("IndexTTS2 Multi-Speaker Voice Synthesis")
    print("=" * 60)
    
    # Initialize Tkinter root window (hidden)
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Get JSON input file
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
    
    # Load JSON to get speaker WAV requirements
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    speaker_files_required = set()
    for entry in entries:
        speaker_files_required.add(entry["voice_sample"])
    
    print(f"\nRequired speaker WAV files from JSON:")
    for spk in sorted(speaker_files_required):
        print(f"  - {spk}")
    
    # Prompt user to select speaker WAV files
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
    
    # Verify all required speaker files exist in the selected directory
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
        print(f"\nError: {len(missing_files)} speaker WAV file(s) not found in selected directory:")
        for mf in missing_files:
            print(f"  - {mf}")
        messagebox.showerror("Missing Files", 
            f"The following speaker files are missing:\n" + "\n".join(missing_files))
        root.destroy()
        return None, None, None, None
    
    print(f"\n✓ All {len(speaker_files_required)} required speaker files found!")
    
    # Get output directory
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
    
    # Get final output filename
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
    
    # Get full path to speaker WAV from speaker_paths mapping
    spk_audio_path = speaker_paths[entry["voice_sample"]]
    
    # Synthesize with TTS2 using parameters from JSON
    tts.infer(
        spk_audio_prompt=spk_audio_path,
        text=entry["text"],
        output_path=out_wav,
        emo_alpha=entry.get("emo_alpha", 0.6),
        use_emo_text=entry.get("use_emo_text", False),
        emo_text=entry.get("emo_text", ""),
        use_random=False,
        verbose=False
    )
    return out_wav

def merge_wavs(wav_paths, output_path):
    """Merge multiple WAV files sequentially"""
    print("\nMerging audio files...")
    merged_audio = []
    samplerate = None
    
    for i, path in enumerate(wav_paths):
        print(f"  Adding segment {i + 1}/{len(wav_paths)}")
        audio, sr = sf.read(path)
        if samplerate is None:
            samplerate = sr
        elif samplerate != sr:
            raise ValueError(f"Sample rate mismatch: {sr} vs {samplerate}")
        merged_audio.append(audio)
    
    merged_audio = np.concatenate(merged_audio)
    sf.write(output_path, merged_audio, samplerate)
    print(f"✓ Merge complete. Output: {output_path}")

def cleanup_temp_files(wav_paths):
    """Remove temporary generated files"""
    print("\nCleaning up temporary files...")
    for path in wav_paths:
        try:
            os.remove(path)
            print(f"  Removed: {path}")
        except Exception as e:
            print(f"  Warning: Could not remove {path}: {e}")

def main():
    """Main execution flow"""
    json_path, output_dir, final_output_path, speaker_paths = get_user_input()
    
    if json_path is None:
        return
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Input JSON: {json_path}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Final Output: {final_output_path}")
    print(f"{'='*60}\n")
    
    # Read JSON input
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    print(f"Processing {len(entries)} entries...\n")
    
    wav_files = []
    # Synthesize each segment
    for idx, entry in enumerate(entries):
        try:
            wav_file = synthesize_entry(entry, idx, output_dir, speaker_paths)
            wav_files.append(wav_file)
        except Exception as e:
            print(f"Error synthesizing entry {idx}: {e}")
            return
    
    # Merge sequentially into one output file
    try:
        merge_wavs(wav_files, final_output_path)
    except Exception as e:
        print(f"Error merging files: {e}")
        return
    
    # Clean up temporary files
    cleanup_response = input("\nDelete temporary segment files? (y/n): ").strip().lower()
    if cleanup_response == 'y':
        cleanup_temp_files(wav_files)
    
    print(f"\n{'='*60}")
    print("✓ Synthesis complete!")
    print(f"Final output saved to: {final_output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
