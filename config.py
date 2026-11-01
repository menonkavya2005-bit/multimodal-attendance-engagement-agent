"""Central settings. Tune these numbers to change how the system behaves."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---- Vision (Milestone 2) ----
MIN_DETECTION_CONF = 0.6        # face detection confidence
PRESENCE_THRESHOLD_SEC = 5.0    # seconds a face must be seen to mark "Present"
LOST_TIMEOUT_SEC = 3.0          # face unseen this long -> track is closed
MATCH_DIST_FRAC = 0.15          # max movement (fraction of frame width) to keep same ID
ATTENTION_YAW_RATIO = 0.35      # lower = stricter "looking at screen" rule

# ---- Audio (Milestone 3) ----
AUDIO_SR = 16000                # sample rate
AUDIO_CHUNK_SEC = 1.0           # analyse audio in 1-second windows
SILENCE_RMS = 0.005             # below this volume = silence
SPEECH_BAND = (200, 3400)       # Hz range where human speech lives
SPEECH_BAND_RATIO = 0.50        # share of energy inside speech band to call it speech
SPEECH_MAX_FLATNESS = 0.20      # speech is "peaky"; noise is "flat"

# ---- Agent (Milestone 4) ----
W_VISION = 0.7                  # weight of attention in engagement score
W_AUDIO = 0.3                   # weight of audio activity
AUDIO_SCORE = {"speech": 1.0, "silence": 0.6, "noise": 0.2, "unavailable": 0.6}
HIGH_ENGAGEMENT = 70
MEDIUM_ENGAGEMENT = 40

# ---- Analytics (Milestone 5) ----
LOW_ATTENTION_STUDENT_PCT = 40  # student flagged below this attention %
