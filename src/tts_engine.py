"""
Local Text-to-Speech Engine using offline SAPI5 (pyttsx3).
Runs 100% locally on CPU without external API calls or internet connection.
"""

import os
import re
import logging
import tempfile
import threading
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Global lock to serialize TTS engine access safely
_tts_lock = threading.Lock()
_current_live_engine = None


def clean_text_for_tts(text: str) -> str:
    """
    Clean scientific paper text to sound natural when spoken aloud.
    Strips markdown symbols, citation brackets like [12], URLs, and LaTeX equations.
    """
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' link ', text)

    # Remove citations like [1], [1, 2], [1-3]
    text = re.sub(r'\[\s*\d+(?:[,\s–-]+\d+)*\s*\]', '', text)

    # Clean markdown formatting (*, _, #, `, ~)
    text = re.sub(r'[*_#`~]', '', text)

    # Clean LaTeX math blocks ($...$)
    text = re.sub(r'\$[^$]+\$', ' equation ', text)

    # Replace multiple newlines or spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_available_voices() -> List[Dict[str, str]]:
    """
    List all offline voices installed on the local system.

    Returns:
        List of dicts with 'id', 'name', and 'gender'.
    """
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass

    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        voice_list = []
        for v in voices:
            name = v.name
            gender = "Female" if "zira" in name.lower() or "female" in getattr(v, 'gender', '').lower() else "Male"
            voice_list.append({
                "id": v.id,
                "name": name,
                "gender": gender
            })
        return voice_list
    except Exception as e:
        logger.warning(f"Failed to query pyttsx3 voices: {e}")
        return [{"id": "default", "name": "Default System Voice", "gender": "Neutral"}]


def synthesize_speech_wav(
    text: str,
    output_path: Optional[str] = None,
    voice_id: Optional[str] = None,
    rate: int = 175,
    max_chars: int = 12000
) -> Tuple[bool, str, bytes]:
    """
    Synthesize text to a WAV audio file on the local machine using pyttsx3.

    Args:
        text: Plain text to synthesize.
        output_path: Destination WAV file path (auto-generated if None).
        voice_id: Specific voice ID (defaults to system default).
        rate: Speaking rate in words per minute (default 175).
        max_chars: Maximum characters to synthesize in one pass.

    Returns:
        Tuple of (success: bool, file_path: str, audio_bytes: bytes)
    """
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        return False, "No text to synthesize", b""

    # Truncate to avoid excessive generation time on entire 30-page documents
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "... This concludes the audio excerpt of the paper."

    if not output_path:
        os.makedirs("downloads/audio", exist_ok=True)
        import hashlib
        h = hashlib.md5(cleaned[:100].encode("utf-8")).hexdigest()[:10]
        output_path = os.path.join("downloads", "audio", f"paper_{h}.wav")

    # If cached file exists and is valid, return immediately
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        with open(output_path, "rb") as f:
            return True, output_path, f.read()

    with _tts_lock:
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)

            if voice_id:
                try:
                    engine.setProperty("voice", voice_id)
                except Exception:
                    pass

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Use a temporary file first to ensure atomic write
            temp_dir = tempfile.gettempdir()
            temp_wav = os.path.join(temp_dir, f"temp_tts_{os.getpid()}_{threading.get_ident()}.wav")

            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

            engine.save_to_file(cleaned, temp_wav)
            engine.runAndWait()

            if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                with open(temp_wav, "rb") as src, open(output_path, "wb") as dst:
                    dst.write(src.read())
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

                with open(output_path, "rb") as f:
                    audio_bytes = f.read()

                return True, output_path, audio_bytes
            else:
                return False, "Failed to generate audio output file", b""

        except Exception as e:
            logger.error(f"Local TTS synthesis error: {e}")
            return False, str(e), b""


def speak_live_background(text: str, voice_id: Optional[str] = None, rate: int = 175):
    """
    Speak text immediately aloud through local speakers in a background thread.
    """
    def _worker():
        global _current_live_engine
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        try:
            import pyttsx3
            cleaned = clean_text_for_tts(text)
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            if voice_id:
                try:
                    engine.setProperty("voice", voice_id)
                except Exception:
                    pass
            _current_live_engine = engine
            engine.say(cleaned)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"Live speech error: {e}")
        finally:
            _current_live_engine = None

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def stop_live_speech():
    """Stop ongoing live speech if running."""
    global _current_live_engine
    if _current_live_engine:
        try:
            _current_live_engine.stop()
        except Exception:
            pass
