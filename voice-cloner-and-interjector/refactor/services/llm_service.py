import requests
import json
import subprocess
from typing import List, Dict, Optional, Any, TypedDict
from config import LLM_CFG

class TriggerData(TypedDict):
    trigger_word: str
    char_pos: int
    category: str
    pos_percent: float

class LLMService:
    """Handles interaction with Ollama for trigger detection and text generation."""

    def ensure_model_loaded(self) -> bool:
        """Checks if Ollama is running and the model is available."""
        try:
            resp = requests.get(LLM_CFG.check_url, timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            
            if LLM_CFG.model not in models:
                print(f"Downloading {LLM_CFG.model}...")
                subprocess.run(["ollama", "pull", LLM_CFG.model], check=True)
            return True
        except Exception as e:
            print(f"LLM Service Error: {e}")
            return False

    def detect_trigger(self, text: str) -> List[TriggerData]:
        """
        Analyzes text to find interjection triggers.
        
        Args:
            text: The speaker's text.
            
        Returns:
            List[TriggerData]: List of potential triggers.
        """
        prompt = self._build_trigger_prompt(text)
        
        try:
            response = requests.post(
                LLM_CFG.api_url,
                json={
                    "model": LLM_CFG.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=LLM_CFG.timeout
            )
            
            if response.status_code == 200:
                result_text = response.json().get("response", "")
                data = json.loads(result_text)
                
                # Validate structure
                if "trigger_word" in data and "char_pos" in data:
                    data["pos_percent"] = data["char_pos"] / len(text) if len(text) > 0 else 0
                    return [data] # Returning list to maintain compatibility
            return []
            
        except Exception as e:
            print(f"Trigger detection failed: {e}")
            return []

    def generate_interjection(self, context_text: str, speaker_persona: str) -> str:
        """Generates a natural interjection based on context."""
        prompt = (
            f"You are {speaker_persona}. The speaker said: \"{context_text[:200]}\"\n"
            "Generate ONE SHORT natural interjection (2-6 words). No quotes."
        )
        
        try:
            response = requests.post(
                LLM_CFG.api_url,
                json={
                    "model": LLM_CFG.model,
                    "prompt": prompt,
                    "stream": False,
                    "num_predict": 20
                },
                timeout=LLM_CFG.timeout
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip().replace('"', '')
        except Exception:
            pass
        
        return "Yeah, I see." # Ultimate fallback

    def _build_trigger_prompt(self, text: str) -> str:
        return f"""
        Analyze the text: "{text}"
        Find the best moment for an interjection (question, strong statement, problem).
        Return JSON ONLY: {{"trigger_word": "word", "char_pos": 12, "category": "statement"}}
        """