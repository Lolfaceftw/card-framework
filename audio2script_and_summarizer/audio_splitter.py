import os
import json
import argparse
import logging
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [SPLITTER] %(message)s', datefmt='%H:%M:%S')

def load_audio(path):
    """Loads audio file using Pydub."""
    logging.info(f"Loading audio: {path}...")
    try:
        return AudioSegment.from_file(path)
    except Exception as e:
        logging.error(f"Failed to load audio: {e}")
        raise e

def get_safe_zone(audio_len_ms):
    """
    Returns (start_ms, end_ms) safe zone.
    Rule: If > 10 mins, exclude first and last minute to avoid intro/outro music.
    """
    TEN_MINUTES_MS = 10 * 60 * 1000
    ONE_MINUTE_MS = 60 * 1000

    if audio_len_ms > TEN_MINUTES_MS:
        start = ONE_MINUTE_MS
        end = audio_len_ms - ONE_MINUTE_MS
        logging.info(f"Audio > 10m. Applying safe zone: {start/1000}s to {end/1000}s")
        return start, end
    else:
        logging.info("Audio < 10m. Using full duration.")
        return 0, audio_len_ms

def extract_speaker_samples(audio, json_data, output_dir):
    """
    Extracts 30s samples for each speaker based on diarization segments.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Group segments by speaker
    segments = json_data.get("segments", [])
    speaker_map = {} 

    safe_start, safe_end = get_safe_zone(len(audio))

    for seg in segments:
        spk = seg["speaker"]
        start = seg["start_time"]
        end = seg["end_time"]

        # Clip segment to safe zone
        valid_start = max(start, safe_start)
        valid_end = min(end, safe_end)

        if valid_end > valid_start:
            if spk not in speaker_map:
                speaker_map[spk] = []
            speaker_map[spk].append((valid_start, valid_end))

    # 2. Process each speaker
    TARGET_DURATION = 30 * 1000 # 30 seconds in ms

    for speaker, intervals in speaker_map.items():
        logging.info(f"Processing {speaker}...")
        
        final_sample = None
        
        # STRATEGY A: Check for a single contiguous chunk >= 30s
        for start, end in intervals:
            duration = end - start
            if duration >= TARGET_DURATION:
                logging.info(f"  -> Found contiguous segment ({duration/1000}s)")
                final_sample = audio[start : start + TARGET_DURATION]
                break
        
        # STRATEGY B: Concatenate smaller chunks if Strategy A failed
        if final_sample is None:
            logging.info(f"  -> No contiguous 30s segment. Concatenating...")
            combined_audio = AudioSegment.empty()
            
            # Sort chunks by length (longest first) to reduce number of cuts
            sorted_intervals = sorted(intervals, key=lambda x: x[1]-x[0], reverse=True)
            
            for start, end in sorted_intervals:
                chunk = audio[start:end]
                combined_audio += chunk
                if len(combined_audio) >= TARGET_DURATION:
                    break
            
            # Trim to exactly 30s (or less if total data is small)
            limit = min(len(combined_audio), TARGET_DURATION)
            final_sample = combined_audio[:limit]

        # 3. Export
        if final_sample and len(final_sample) > 0:
            out_path = os.path.join(output_dir, f"{speaker}.wav")
            final_sample.export(out_path, format="wav")
            logging.info(f"  -> Saved {out_path} ({len(final_sample)/1000}s)")
        else:
            logging.warning(f"  -> Could not extract any audio for {speaker}!")

def main():
    parser = argparse.ArgumentParser(description="CARD Audio Splitter")
    parser.add_argument("--audio", required=True, help="Path to original audio file")
    parser.add_argument("--json", required=True, help="Path to diarization JSON")
    parser.add_argument("--output-dir", required=True, help="Directory to save speaker samples")
    
    args = parser.parse_args()

    try:
        audio = load_audio(args.audio)
        with open(args.json, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        extract_speaker_samples(audio, json_data, args.output_dir)
    except Exception as e:
        logging.error(f"Process failed: {e}")

if __name__ == "__main__":
    main()