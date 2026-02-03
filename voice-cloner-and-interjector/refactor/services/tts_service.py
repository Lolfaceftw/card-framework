import os
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from indextts.infer_v2 import IndexTTS2
# Mock import for structure - in production, this is: from indextts.infer_v2 import IndexTTS2
try:
    from indextts.infer_v2 import IndexTTS2 as _IndexTTS2Impl
    _TTS_AVAILABLE = True
except ImportError:
    _IndexTTS2Impl = None
    _TTS_AVAILABLE = False

from config import TTS_CFG
    
class TTSService:
    """Wrapper for IndexTTS2 to manage initialization and inference."""
    
    def __init__(self):
        self.model: Optional["IndexTTS2"] = None

    def load_model(self):
        """Lazy loads the TTS model to GPU."""
        if self.model is None:
            if IndexTTS2 is None:
                if not _TTS_AVAILABLE or _IndexTTS2Impl is None:
                    raise ImportError("IndexTTS2 library not found.")
            
            print("Loading IndexTTS2 Model...")
            self.model = _IndexTTS2Impl(
                cfg_path=TTS_CFG.config_path,
                model_dir=TTS_CFG.model_dir,
                device=TTS_CFG.device,
                use_fp16=TTS_CFG.use_fp16,
                use_cuda_kernel=True
            )

    def synthesize(self, text: str, speaker_wav: str, output_path: str, 
                  emo_text: str = "", emo_alpha: float = 0.6) -> str:
        """
        Runs inference.
        
        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            self.load_model()
            
        self.model.infer(
            spk_audio_prompt=speaker_wav,
            text=text,
            output_path=output_path,
            emo_alpha=emo_alpha,
            use_emo_text=bool(emo_text),
            emo_text=emo_text,
            use_random=False,
            verbose=False
        )
        return output_path