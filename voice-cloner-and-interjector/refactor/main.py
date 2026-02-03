import argparse
import json
import os
import sys
from core.pipeline import CardPipeline

def main():
    parser = argparse.ArgumentParser(description="CARD: Constraint-aware Audio Resynthesis")
    parser.add_argument("--input_json", required=True, help="Path to conversation JSON")
    parser.add_argument("--speaker_dir", required=True, help="Directory with speaker WAVs")
    parser.add_argument("--output_dir", required=True, help="Directory for output")
    parser.add_argument("--filename", default="podcast.wav", help="Output filename")
    
    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.input_json):
        print(f"Error: JSON file not found: {args.input_json}")
        sys.exit(1)

    with open(args.input_json, "r") as f:
        entries = json.load(f)

    # Map speakers
    speaker_files = {e["voice_sample"] for e in entries}
    speaker_paths = {}
    for spk in speaker_files:
        path = os.path.join(args.speaker_dir, spk)
        if not os.path.exists(path):
            print(f"Error: Speaker file missing: {path}")
            sys.exit(1)
        speaker_paths[spk] = path

    # Run Pipeline
    pipeline = CardPipeline()
    try:
        pipeline.run(entries, speaker_paths, args.output_dir, args.filename)
    except Exception as e:
        print(f"Critical Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()