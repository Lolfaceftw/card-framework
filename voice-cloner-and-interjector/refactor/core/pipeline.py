import os
import random
import json
from typing import List, Dict
from services.llm_service import LLMService
from services.tts_service import TTSService
from services.audio_service import AudioService
from config import AUDIO_CFG

class CardPipeline:
    """
    Constraint-aware Audio Resynthesis and Distillation (CARD) Pipeline.
    Orchestrates the flow between LLM, TTS, and Audio processing.
    """
    
    def __init__(self):
        self.llm = LLMService()
        self.tts = TTSService()
        self.audio = AudioService()
        self.interjection_log = []

    def run(self, entries: List[Dict], speaker_paths: Dict[str, str], output_dir: str, output_filename: str):
        """
        Main execution loop.
        """
        if not self.llm.ensure_model_loaded():
            raise RuntimeError("LLM not available.")

        processed_segments = []

        for idx, entry in enumerate(entries):
            print(f"Processing segment {idx+1}/{len(entries)}...")
            
            # 1. Synthesize Main Speaker
            main_wav_path = os.path.join(output_dir, f"seg_{idx}.wav")
            self.tts.synthesize(
                text=entry["text"],
                speaker_wav=speaker_paths[entry["voice_sample"]],
                output_path=main_wav_path,
                emo_text=entry.get("emo_text", ""),
                emo_alpha=entry.get("emo_alpha", 0.6)
            )
            
            main_audio = self.audio.load_audio(main_wav_path)
            
            # 2. Determine Interjection (Logic from abstract)
            if idx > 0 and random.random() < AUDIO_CFG.interjection_prob:
                main_audio = self._handle_interjection(
                    idx, entry, main_audio, speaker_paths, output_dir
                )

            processed_segments.append(main_audio)

        # 3. Merge
        print("Merging segments...")
        final_audio = self.audio.merge_segments(processed_segments)
        final_path = os.path.join(output_dir, output_filename)
        final_audio.export(final_path, format="wav")
        
        # 4. Save Log
        self._save_log(final_path)
        print(f"Done! Output at: {final_path}")

    def _handle_interjection(self, idx: int, entry: Dict, main_audio, 
                             speaker_paths: Dict, output_dir: str):
        """Sub-routine for calculating and generating interjections."""
        
        # A. Detect Trigger
        triggers = self.llm.detect_trigger(entry["text"])
        if not triggers:
            return main_audio

        trigger = random.choice(triggers)
        
        # B. Calculate Timing
        timing = self.audio.calculate_timing(
            entry["text"], main_audio, trigger["pos_percent"]
        )

        # C. Select Interjector (Different from current speaker)
        others = [k for k in speaker_paths.keys() if k != entry["voice_sample"]]
        if not others:
            return main_audio
            
        interjector_name = random.choice(others)
        interjector_path = speaker_paths[interjector_name]

        # D. Generate Text & Audio
        int_text = self.llm.generate_interjection(entry["text"], "Listener")
        int_wav_path = os.path.join(output_dir, f"int_{idx}.wav")
        
        self.tts.synthesize(
            text=int_text,
            speaker_wav=interjector_path,
            output_path=int_wav_path,
            emo_text="Casual listening agreement",
            emo_alpha=0.5
        )
        
        # E. Mix
        int_audio = self.audio.load_audio(int_wav_path)
        mixed_audio = self.audio.overlay(main_audio, int_audio, timing["final_pos_ms"])
        
        # Log
        self.interjection_log.append({
            "segment": idx,
            "trigger": trigger["trigger_word"],
            "interjection": int_text,
            "timing": timing
        })
        
        return mixed_audio

    def _save_log(self, final_path: str):
        log_path = final_path.replace(".wav", "_log.json")
        with open(log_path, "w") as f:
            json.dump(self.interjection_log, f, indent=2)