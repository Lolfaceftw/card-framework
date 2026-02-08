import json
import os
import argparse
import typing
import logging
from typing import List
import google.generativeai as genai
from pydantic import BaseModel, Field

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Data Structures (FIXED for Gemini) ---
class DialogueLine(BaseModel):
    speaker: str = Field(..., description="The speaker label (e.g., SPEAKER_00).")
    text: str = Field(..., description="The summarized conversational line (10-15 seconds).")
    emo_text: str = Field(..., description="Emotion description (e.g., 'Warm', 'Shocked').")
    emo_alpha: float = Field(..., description="Intensity (0.5 to 0.9).") 

class PodcastScript(BaseModel):
    dialogue: List[DialogueLine]

# --- Core Logic ---

def load_transcript(input_data: str) -> str:
    """Loads JSON string OR file path and converts to text block."""
    if input_data.strip().startswith("{") or input_data.strip().startswith("["):
        data = json.loads(input_data)
    else:
        with open(input_data, 'r', encoding='utf-8') as f:
            data = json.load(f)

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

def generate_summary(transcript_text: str, api_key: str) -> dict:
    """Sends transcript to Gemini 1.5 Flash and forces JSON output."""
    
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
        "response_schema": PodcastScript
    }

    model = genai.GenerativeModel(
        model_name="gemini-flash-latest",
        generation_config=generation_config,
        system_instruction="""
        You are an expert podcast editor. 
        Rewrite the transcript into a concise, engaging summary script (approx 1/5th length).
        
        CONSTRAINTS:
        1. Keep it conversational and natural (short sentences).
        2. Use exact speaker labels (SPEAKER_00, etc.) from the input.
        3. Annotate every line with 'emo_text' describing the tone.
        4. Each line should be ~10-15 seconds of speech.
        5. 'emo_alpha' must be a float between 0.5 and 0.9.
        """
    )

    logger.info("Sending transcript to Gemini 1.5 Flash...")
    
    try:
        response = model.generate_content(f"Here is the raw transcript:\n\n{transcript_text}")
        json_output = json.loads(response.text)
        return json_output
        
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        raise e

def post_process_script(script_data: dict, voice_sample_dir: str) -> List[dict]:
    """Injects file paths."""
    final_output = []
    
    dialogue_list = script_data.get('dialogue', [])

    for line in dialogue_list:
        # Flexible handling if line is dict (JSON) or Pydantic object
        if isinstance(line, dict):
            spk = line.get('speaker')
            txt = line.get('text')
            emo = line.get('emo_text')
            alpha = line.get('emo_alpha', 0.6) # We apply default here safely
        else:
            spk = line.speaker
            txt = line.text
            emo = line.emo_text
            alpha = getattr(line, 'emo_alpha', 0.6)

        voice_path = os.path.join(voice_sample_dir, f"{spk}.wav")

        entry = {
            "speaker": spk,
            "voice_sample": voice_path.replace("\\", "/"),
            "use_emo_text": True,
            "emo_text": emo,
            "emo_alpha": alpha,
            "text": txt
        }
        final_output.append(entry)

    return final_output

# --- CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="CARD Script Summarizer (Gemini Edition)")
    parser.add_argument("--transcript", required=True, help="Path to input JSON transcript")
    parser.add_argument("--voice-dir", required=True, help="Directory for voice samples")
    parser.add_argument("--output", default="summarized_script.json", help="Output path")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"), help="Google API Key")

    args = parser.parse_args()

    if not args.api_key:
        logger.error("No API Key found. Set GEMINI_API_KEY or pass --api-key.")
        return

    logger.info(f"Loading: {args.transcript}")
    raw_text = load_transcript(args.transcript)

    try:
        structured_data = generate_summary(raw_text, args.api_key)
        final_json = post_process_script(structured_data, args.voice_dir)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=2)
            
        logger.info(f"Success! Output: {args.output}")
        
    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()