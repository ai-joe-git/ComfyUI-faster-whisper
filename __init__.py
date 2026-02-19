# __init__.py (complete)
# Put this file in: ComfyUI/custom_nodes/ComfyUI-faster-whisper/__init__.py
# (i.e., replace the existing __init__.py in that repo)

from .nodes import *

# ----------------------------
# Glue node: AUDIO -> FILEPATH (temp WAV)
# ----------------------------
import os
import time
import wave
import tempfile

class AudioToWavFilepath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "prefix": ("STRING", {"default": "stt_audio"}),
            }
        }

    RETURN_TYPES = ("FILEPATH",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "run"
    CATEGORY = "FasterWhisper"

    def run(self, audio, prefix="stt_audio"):
        # ComfyUI AUDIO usually: {"waveform": tensor/ndarray, "sample_rate": int}
        wf = audio.get("waveform", None)
        sr = int(audio.get("sample_rate", 44100))

        # Some variants may use "samples"
        if wf is None:
            wf = audio.get("samples", None)

        if wf is None:
            raise ValueError("AUDIO input missing 'waveform' (or 'samples').")

        # Convert to numpy
        try:
            wf_np = wf.detach().cpu().numpy()
        except Exception:
            wf_np = wf

        import numpy as np
        wf_np = np.array(wf_np)
        wf_np = np.squeeze(wf_np)

        # Handle common shapes:
        # - (samples,)
        # - (channels, samples)
        # - (batch, channels, samples) -> take batch[0]
        if wf_np.ndim == 1:
            wf_np = wf_np[None, :]          # (1, samples)
        elif wf_np.ndim == 3:
            wf_np = wf_np[0]                # (channels, samples)

        # Clamp float waveform to int16 PCM
        wf_np = np.clip(wf_np, -1.0, 1.0)
        pcm = (wf_np * 32767.0).astype(np.int16)

        out_dir = os.path.join(tempfile.gettempdir(), "comfyui_stt")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{prefix}_{int(time.time()*1000)}.wav")

        # Write WAV (channels interleaved)
        with wave.open(out_path, "wb") as f:
            f.setnchannels(int(pcm.shape[0]))
            f.setsampwidth(2)               # int16
            f.setframerate(sr)
            f.writeframes(pcm.T.tobytes())

        return (out_path,)


# ----------------------------
# Glue node: TRANSCRIPTIONS -> STRING (plain text)
# ----------------------------
class TranscriptionsToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transcriptions": ("TRANSCRIPTIONS",),
                "separator": ("STRING", {"default": " "}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "FasterWhisper"

    def run(self, transcriptions, separator=" "):
        parts = []
        for seg in transcriptions:
            t = (seg.get("text") or "").strip()
            if t:
                parts.append(t)
        return (separator.join(parts),)


NODE_CLASS_MAPPINGS = {
    "LoadFasterWhisperModel": LoadFasterWhisperModel,
    "FasterWhisperTranscription": FasterWhisperTranscription,
    "FasterWhisperToSubtitle": FasterWhisperToSubtitle,
    "SaveSubtitle": SaveSubtitle,
    "InputFilePath": InputFilePath,

    # NEW
    "AudioToWavFilepath": AudioToWavFilepath,
    "TranscriptionsToText": TranscriptionsToText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFasterWhisperModel": "(Down)Load FasterWhisper Model",
    "FasterWhisperTranscription": "FasterWhisper Transcription",
    "FasterWhisperToSubtitle": "FasterWhisper To Subtitle",
    "SaveSubtitle": "Save Subtitle",
    "InputFilePath": "Input FilePath",

    # NEW
    "AudioToWavFilepath": "STT: AUDIO → WAV Filepath",
    "TranscriptionsToText": "STT: Transcriptions → Text",
}

# Temporal fix of the bug : https://github.com/jhj0517/Whisper-WebUI/issues/144
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
