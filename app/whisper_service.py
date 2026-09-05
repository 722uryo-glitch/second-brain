import tempfile
from pathlib import Path
from .config import WHISPER_MODEL

_model = None


def _get_model():
    global _model

    import whisper

    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL)

    return _model


def transcribe_bytes(data: bytes, suffix: str = ".wav") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name

    try:
        result = _get_model().transcribe(path)
        return result.get("text", "").strip()
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass