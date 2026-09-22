"""
Local Text-to-Speech Engine using offline SAPI5 (pyttsx3).
Runs 100% locally on CPU without external API calls or internet connection.
"""

import os
import re
import logging
import tempfile
import threading
import subprocess
import sys
import wave
import io
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Global lock to serialize TTS engine access safely
_tts_lock = threading.Lock()
_current_live_engine = None


def is_playable_wav_bytes(audio_bytes: bytes) -> bool:
    """Validate the exact payload that Streamlit will send to the browser."""
    if not audio_bytes or len(audio_bytes) < 44:
        return False
    try:
        if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            return False
        with wave.open(io.BytesIO(audio_bytes), "rb") as audio:
            return audio.getcomptype() == "NONE" and audio.getnframes() > 0
    except (EOFError, wave.Error):
        return False


def _is_pcm_wav(path: str) -> bool:
    """Return whether ``path`` is a non-empty browser-decodable PCM WAV."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return False
        with open(path, "rb") as handle:
            return is_playable_wav_bytes(handle.read())
    except (OSError, wave.Error):
        return False


def _convert_to_pcm_wav(source_path: str, output_path: str) -> bool:
    """Convert macOS AIFF-C/AIFF output to standard 16-bit PCM WAV."""
    if sys.platform != "darwin":
        # Windows SAPI5 and Linux speech backends are asked to emit WAV
        # directly below. ``afconvert`` is an Apple-only executable.
        return False
    converted_path = f"{output_path}.converted.wav"
    try:
        if os.path.exists(converted_path):
            os.remove(converted_path)
        result = subprocess.run(
            ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@22050", source_path, converted_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not _is_pcm_wav(converted_path):
            logger.error("WAV conversion failed: %s", result.stderr.strip())
            return False
        os.replace(converted_path, output_path)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Could not convert speech audio to WAV: %s", exc)
        return False
    finally:
        if os.path.exists(converted_path):
            try:
                os.remove(converted_path)
            except OSError:
                pass


def _synthesize_with_macos_say(text: str, temp_aiff: str, voice_id: Optional[str], rate: int) -> None:
    """Native fallback when pyttsx3 is unavailable in the Streamlit environment."""
    command = ["/usr/bin/say", "-o", temp_aiff, "-r", str(rate)]
    if voice_id and voice_id != "default":
        command.extend(["-v", voice_id])
    command.append(text)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode and voice_id and voice_id != "default":
        # pyttsx3 voice IDs are not always accepted by the ``say`` utility.
        # Keep speech working by falling back to the selected system default.
        subprocess.run(["/usr/bin/say", "-o", temp_aiff, "-r", str(rate), text], check=True, capture_output=True, text=True)
    else:
        result.check_returncode()


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
        # Include the complete spoken text and synthesis settings. The previous
        # first-100-character key could reuse unrelated or stale audio when a
        # paper shared the same opening text, voice settings changed, or the
        # encoding implementation was upgraded.
        cache_material = "\0".join([
            "pcm-wav-v2",
            voice_id or "default",
            str(rate),
            cleaned,
        ])
        cache_hash = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()[:20]
        output_path = os.path.join("downloads", "audio", f"paper_{cache_hash}.wav")

    # Legacy macOS pyttsx3 files may have an AIFF-C payload despite their .wav
    # extension. Repair those cached files before returning them to st.audio.
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        if _is_pcm_wav(output_path):
            with open(output_path, "rb") as handle:
                return True, output_path, handle.read()
        _convert_to_pcm_wav(output_path, output_path)
        if _is_pcm_wav(output_path):
            with open(output_path, "rb") as handle:
                return True, output_path, handle.read()
        # An old cache can contain an empty AIFF-C container. Discard it and
        # synthesize fresh speech instead of returning silent audio.
        try:
            os.remove(output_path)
        except OSError:
            pass

    with _tts_lock:
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # macOS NSSpeechSynthesizer writes AIFF-C even when a path ends in
            # .wav. Windows SAPI5 and typical Linux pyttsx3 backends write WAV
            # directly, so only macOS needs an AIFF staging file.
            temp_dir = tempfile.gettempdir()
            temp_suffix = ".aiff" if sys.platform == "darwin" else ".wav"
            temp_audio = os.path.join(temp_dir, f"temp_tts_{os.getpid()}_{threading.get_ident()}{temp_suffix}")

            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except Exception:
                    pass

            if sys.platform == "darwin":
                _synthesize_with_macos_say(cleaned, temp_audio, voice_id, rate)
            else:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", rate)
                if voice_id:
                    engine.setProperty("voice", voice_id)
                engine.save_to_file(cleaned, temp_audio)
                engine.runAndWait()

            if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                if sys.platform == "darwin":
                    if not _convert_to_pcm_wav(temp_audio, output_path) or not _is_pcm_wav(output_path):
                        return False, "Failed to convert generated speech to PCM WAV", b""
                elif _is_pcm_wav(temp_audio):
                    os.replace(temp_audio, output_path)
                else:
                    return False, "Speech engine did not generate a playable PCM WAV file", b""
                try:
                    os.remove(temp_audio)
                except Exception:
                    pass

                with open(output_path, "rb") as handle:
                    audio_bytes = handle.read()

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
