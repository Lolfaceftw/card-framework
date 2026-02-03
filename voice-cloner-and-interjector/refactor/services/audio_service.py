import random
from typing import Dict, Any, Optional
from pydub import AudioSegment
from config import AUDIO_CFG

class AudioService:
    """Handles audio manipulation and timing calculations."""

    def load_audio(self, path: str) -> AudioSegment:
        return AudioSegment.from_wav(path)

    def calculate_timing(self, text: str, audio: AudioSegment, 
                        trigger_percent: float) -> Dict[str, int]:
        """
        Calculates the millisecond timestamp for the interjection.
        """
        duration_ms = len(audio)
        trigger_ms = int(duration_ms * trigger_percent)
        
        reaction_delay = random.randint(
            AUDIO_CFG.min_reaction_delay_ms, 
            AUDIO_CFG.max_reaction_delay_ms
        )
        
        position_ms = trigger_ms + reaction_delay
        
        # Clamp to audio bounds (don't start after audio ends)
        position_ms = max(0, min(position_ms, duration_ms - 500))
        
        return {
            "trigger_ms": trigger_ms,
            "reaction_delay": reaction_delay,
            "final_pos_ms": position_ms,
            "duration_ms": duration_ms
        }

    def overlay(self, main: AudioSegment, overlay: AudioSegment, 
                position_ms: int) -> AudioSegment:
        return main.overlay(overlay, position=position_ms)

    def merge_segments(self, segments: list[AudioSegment]) -> AudioSegment:
        combined = segments[0]
        pause = AudioSegment.silent(duration=AUDIO_CFG.pause_ms)
        
        for seg in segments[1:]:
            combined = combined.append(pause, crossfade=0)
            combined = combined.append(seg, crossfade=AUDIO_CFG.crossfade_ms)
            
        return combined