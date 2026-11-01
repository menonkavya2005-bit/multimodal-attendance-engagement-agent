"""Milestone 3: audio signal extraction and noise / speech activity analysis."""
import threading
import time
import numpy as np
import librosa
from config import (AUDIO_SR, AUDIO_CHUNK_SEC, SILENCE_RMS, SPEECH_BAND,
                    SPEECH_BAND_RATIO, SPEECH_MAX_FLATNESS)

try:
    import sounddevice as sd
except Exception:  # missing PortAudio / no device
    sd = None

UNAVAILABLE = {"state": "unavailable", "rms": 0.0, "speech_ratio": 0.0, "flatness": 0.0}


# ---------- Activity 3.2: noise and speech analysis ----------
def analyze_chunk(y, sr=AUDIO_SR):
    """Classify one audio window as 'silence', 'speech' or 'noise'."""
    y = np.asarray(y, dtype=np.float32).flatten()
    if y.size < 512:
        return {"state": "silence", "rms": 0.0, "speech_ratio": 0.0, "flatness": 0.0}
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < SILENCE_RMS:
        return {"state": "silence", "rms": rms, "speech_ratio": 0.0, "flatness": 0.0}

    mag = np.abs(librosa.stft(y, n_fft=512, hop_length=256))
    power = mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    band = (freqs >= SPEECH_BAND[0]) & (freqs <= SPEECH_BAND[1])
    speech_ratio = float(power[band].sum() / (power.sum() + 1e-12))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=mag)))

    is_speech = speech_ratio >= SPEECH_BAND_RATIO and flatness <= SPEECH_MAX_FLATNESS
    return {"state": "speech" if is_speech else "noise", "rms": rms,
            "speech_ratio": speech_ratio, "flatness": flatness}


# ---------- Activity 3.1: audio signal extraction ----------
class AudioMonitor:
    """Live microphone input, analysed once per chunk in a background thread."""

    def __init__(self):
        self._latest = dict(UNAVAILABLE)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        if sd is None:
            print("[audio] sounddevice not available - continuing without audio.")
            return False
        try:
            sd.check_input_settings(samplerate=AUDIO_SR, channels=1)
        except Exception as e:
            print(f"[audio] no usable microphone ({e}) - continuing without audio.")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        n = int(AUDIO_SR * AUDIO_CHUNK_SEC)
        while self._running:
            try:
                y = sd.rec(n, samplerate=AUDIO_SR, channels=1, dtype="float32")
                sd.wait()
                res = analyze_chunk(y[:, 0])
            except Exception:
                res = dict(UNAVAILABLE)
            with self._lock:
                self._latest = res

    def latest(self, elapsed=None):
        with self._lock:
            return dict(self._latest)

    def stop(self):
        self._running = False


class AudioFileSource:
    """Same interface as AudioMonitor, but reads a recorded audio file."""

    def __init__(self, path):
        y, _ = librosa.load(path, sr=AUDIO_SR, mono=True)
        n = int(AUDIO_SR * AUDIO_CHUNK_SEC)
        self.results = [analyze_chunk(y[i:i + n]) for i in range(0, len(y), n)]

    def start(self):
        return len(self.results) > 0

    def latest(self, elapsed=0.0):
        i = int(elapsed / AUDIO_CHUNK_SEC)
        return dict(self.results[i]) if 0 <= i < len(self.results) else dict(UNAVAILABLE)

    def stop(self):
        pass
