"""Milestone 4: multimodal signal fusion and rule-based engagement scoring."""
import os
import pandas as pd
from config import (W_VISION, W_AUDIO, AUDIO_SCORE, HIGH_ENGAGEMENT, MEDIUM_ENGAGEMENT)


class MultimodalAgent:
    def __init__(self):
        self.log = []

    # ---------- Activity 4.2: engagement scoring ----------
    @staticmethod
    def score(faces, attentive, audio_state):
        if faces == 0:
            return 0.0, "No participants"
        attention = attentive / faces
        audio = AUDIO_SCORE.get(audio_state, 0.6)
        value = round(100 * (W_VISION * attention + W_AUDIO * audio), 1)
        if value >= HIGH_ENGAGEMENT:
            level = "High"
        elif value >= MEDIUM_ENGAGEMENT:
            level = "Medium"
        else:
            level = "Low"
        return value, level

    # ---------- Activity 4.1: multimodal fusion ----------
    def step(self, vision, audio, elapsed):
        """Combine one vision reading and one audio reading into a record."""
        value, level = self.score(vision["faces"], vision["attentive"], audio["state"])
        rec = {
            "elapsed_sec": round(elapsed, 1),
            "faces": vision["faces"],
            "attentive": vision["attentive"],
            "audio_state": audio["state"],
            "volume": round(audio.get("rms", 0.0), 4),
            "engagement": value,
            "level": level,
        }
        self.log.append(rec)
        return rec

    def dataframe(self):
        return pd.DataFrame(self.log, columns=[
            "elapsed_sec", "faces", "attentive", "audio_state", "volume", "engagement", "level"])
