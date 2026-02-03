import os
from dataclasses import dataclass

@dataclass
class LLMConfig:
    api_url: str = "http://localhost:11434/api/generate"
    check_url: str = "http://localhost:11434/api/tags"
    model: str = "mistral:7b-instruct-q4_0"
    timeout: int = 30

@dataclass
class TTSConfig:
    config_path: str = "checkpoints/config.yaml"
    model_dir: str = "checkpoints"
    device: str = "cuda:0"
    use_fp16: bool = True

@dataclass
class AudioConfig:
    interjection_prob: float = 0.6
    min_reaction_delay_ms: int = 300
    max_reaction_delay_ms: int = 800
    crossfade_ms: int = 300
    pause_ms: int = 400

# Global instances
LLM_CFG = LLMConfig()
TTS_CFG = TTSConfig()
AUDIO_CFG = AudioConfig()