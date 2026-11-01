# Multimodal Attendance & Engagement Agent

A local, offline AI agent that marks **attendance** from a webcam/video and estimates
**classroom engagement** by fusing vision (MediaPipe + OpenCV) and audio (Librosa) signals
with rule-based reasoning. It then produces teaching suggestions and per-student analytics,
with an optional Streamlit dashboard.

## Features
- Face detection and tracking with anonymous student IDs (no faces are stored)
- Presence-based attendance (a student is marked Present after N seconds on camera)
- Attention estimate from head orientation (looking at the screen or not)
- Audio analysis: silence / speech / background noise
- Multimodal fusion into an engagement score (0-100: High / Medium / Low)
- Teaching suggestions and student analytics (CSV + JSON reports)
- Streamlit dashboard and an automated validation script

## Project structure
```
main.py                     core application (Milestone 6.1)
app.py                      Streamlit dashboard (6.2)
config.py                   all tunable thresholds
vision/vision_module.py     face detection, tracking, attendance (M2)
audio/audio_module.py       audio extraction, noise / speech analysis (M3)
agent/multimodal_agent.py   fusion and engagement scoring (M4)
analytics/analytics.py      suggestions and student analytics (M5)
tests/validate.py           edge-case validation (6.3)
data/                       generated reports
```

## Setup (Windows, Python 3.11)
```
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```
python main.py                                   # webcam + microphone
python main.py --no-audio                        # webcam only
python main.py --video class.mp4 --audio-file class.wav
streamlit run app.py                             # dashboard (after a session)
python tests/validate.py                         # run validation checks
```
Press `q` in the video window to end the session and generate reports.

## How engagement is scored
`engagement = 100 * (0.7 * attention_ratio + 0.3 * audio_score)`
where `attention_ratio` = students facing the camera / students visible, and
`audio_score` is speech 1.0, silence 0.6, noise 0.2. Weights are in `config.py`.

## Limitations
- Student IDs are anonymous; someone who leaves for more than 3 s and returns gets a new ID.
- Attention is estimated from head direction only, not eye gaze.
- Speech vs noise uses simple signal rules; tune thresholds in `config.py` for your room.
