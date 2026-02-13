try:
    from .msdd.msdd import MSDDDiarizer as _MSDDDiarizer
except ImportError:
    MSDDDiarizer = None  # NeMo not available
else:
    MSDDDiarizer = _MSDDDiarizer

try:
    from .pyannote import PyannoteDiarizer as _PyannoteDiarizer
except ImportError:
    PyannoteDiarizer = None  # pyannote not available
else:
    PyannoteDiarizer = _PyannoteDiarizer

__all__ = ["MSDDDiarizer", "PyannoteDiarizer"]
