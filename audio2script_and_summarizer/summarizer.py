import json
import os
import argparse
from typing import List, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Data Structures for Strict JSON Output ---
class DialogueLine(BaseModel):
    speaker: str = Field(..., description="The speaker label (e.g., SPEAKER_00, SPEAKER_01).")
    text: str = Field(..., description="The summarized conversational line (10-15 seconds long).")
    emo_text: str = Field(..., description="A short, natural-language description of the emotion (e.g., 'excited and fast-paced', 'thoughtful and slow').")
    emo_alpha: float = Field(0.6, description="Intensity of the emotion (0.5 to 0.9).")
    # voice_sample will be injected post-generation based on the speaker ID

class PodcastScript(BaseModel):
    dialogue: List[DialogueLine]

# --- Core Logic ---

def load_transcript(input_data: str) -> str:
    """Loads JSON string OR file path and converts to text block."""
    if input_data.strip().startswith("{") or input_data.strip().startswith("["):
        # It's a JSON string already
        data = json.loads(input_data)
    else:
        # It's a file path
        with open(input_data, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Handle the structure mismatch here or in the orchestrator
    # If the orchestrator sent {"segments": [...]}, we use that.
    # If it's a list, we wrap it.
    if isinstance(data, list):
        segments = data
    else:
        segments = data.get('segments', [])

    full_text = ""
    for seg in segments:
        speaker = seg.get('speaker', 'Unknown')
        text = seg.get('text', '').strip()
        full_text += f"[{speaker}]: {text}\n"

    return full_text

def generate_summary(transcript_text: str, api_key: str) -> PodcastScript:
    """Sends transcript to LLM and forces structured JSON output."""
    client = OpenAI(api_key=api_key)

    system_prompt = """
    You are an expert podcast editor and creative writer for the CARD (Constraint-aware Audio Resynthesis) project.

    YOUR GOAL:
    Take a raw transcript and rewrite it into a highly engaging, concise podcast summary that preserves the original "vibe" and information but reduces the length to roughly 1/5th of the original.

    CONSTRAINTS:
    1. **Format**: Output must be a strictly formatted JSON array.
    2. **Dialogue**: Keep the conversation natural. Use short sentences. Avoid "summary language" (e.g., do not say "The speaker discusses..."). Instead, write the actual dialogue they would say.
    3. **Emotions**: You must annotate every line with `emo_text`. Describe the tone, pace, and feeling (e.g., "Warm welcoming," "Shocked and high-pitched," "Skeptical").
    4. **Speaker Consistency**: Use the exact speaker labels (SPEAKER_00, SPEAKER_01) found in the input.
    5. **Duration**: Each line of dialogue should represent about 10-15 seconds of speech.
    """

    logger.info("Sending transcript to LLM for summarization and emotion annotation...")

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06", # Or gpt-4-turbo. High intelligence required for emotion inference.
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the raw transcript:\n\n{transcript_text}"},
        ],
        response_format=PodcastScript,
    )

    return completion.choices[0].message.parsed

def post_process_script(script: PodcastScript, voice_sample_dir: str) -> List[dict]:
    """Injects file paths and ensures final JSON schema compliance."""
    final_output = []

    for line in script.dialogue:
        # Construct the path to the reference audio extracted in Phase 1
        # Assumes file naming convention: output_dir/SPEAKER_XX.wav
        voice_path = os.path.join(voice_sample_dir, f"{line.speaker}.wav")

        # Build the final dict
        entry = {
            "speaker": line.speaker,
            "voice_sample": voice_path.replace("\\", "/"), # Ensure forward slashes
            "use_emo_text": True,
            "emo_text": line.emo_text,
            "emo_alpha": line.emo_alpha,
            "text": line.text
        }
        final_output.append(entry)

    return final_output

# --- CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="CARD Script Summarizer & Emotion Annotator")
    parser.add_argument("--transcript", required=True, help="Path to input WhisperX JSON transcript")
    parser.add_argument("--voice-dir", required=True, help="Directory where separated speaker audios are stored")
    parser.add_argument("--output", default="summarized_script.json", help="Path to save output JSON")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API Key")

    args = parser.parse_args()

    if not args.api_key:
        logger.error("No API Key provided. Set OPENAI_API_KEY env var or pass --api-key.")
        return

    # 1. Load
    logger.info(f"Loading transcript from {args.transcript}")
    raw_text = load_transcript(args.transcript)

    # 2. Generate (LLM)
    try:
        structured_script = generate_summary(raw_text, args.api_key)
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        return

    # 3. Post-Process (Add paths)
    final_json = post_process_script(structured_script, args.voice_dir)

    # 4. Save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2)

    logger.info(f"Success! Summarized script saved to {args.output}")

if __name__ == "__main__":
    main()