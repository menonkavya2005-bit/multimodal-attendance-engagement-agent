"""Milestone 6 (Activity 6.3): validate outputs and edge cases.  Run: python tests/validate.py"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio.audio_module import analyze_chunk
from agent.multimodal_agent import MultimodalAgent
from analytics.analytics import teaching_suggestions, student_analytics, save_reports
from vision.vision_module import FaceTracker

sr = 16000
t = np.arange(sr) / sr
passed = 0


def check(name, cond):
    global passed
    print(("PASS  " if cond else "FAIL  ") + name)
    assert cond, name
    passed += 1


# ---- audio ----
check("silence detected", analyze_chunk(np.zeros(sr))["state"] == "silence")
noise = np.random.RandomState(0).randn(sr) * 0.1
check("white noise detected as noise", analyze_chunk(noise)["state"] == "noise")
f0 = 120 + 10 * np.sin(2 * np.pi * 3 * t)  # wobbling pitch like a real voice
ph = 2 * np.pi * np.cumsum(f0) / sr
voice = sum(np.sin(k * ph) * np.exp(-((k * 120 - 800) / 600) ** 2) for k in range(1, 25)) * 0.1
check("harmonic voice-like signal detected as speech", analyze_chunk(voice)["state"] == "speech")
check("tiny/empty chunk handled", analyze_chunk(np.zeros(5))["state"] == "silence")

# ---- agent ----
check("no faces -> score 0", MultimodalAgent.score(0, 0, "speech")[0] == 0)
check("all attentive + speech -> High", MultimodalAgent.score(3, 3, "speech")[1] == "High")
check("nobody attentive + noise -> Low", MultimodalAgent.score(3, 0, "noise")[1] == "Low")
check("missing audio still works", MultimodalAgent.score(2, 2, "unavailable")[1] == "High")

# ---- analytics ----
check("empty log gives a message", len(teaching_suggestions(pd.DataFrame())) == 1)
check("empty students ok", student_analytics([]).empty)
agent = MultimodalAgent()
for i in range(20):
    agent.step({"faces": 3, "attentive": 3 if i < 10 else 0}, {"state": "noise", "rms": 0.1}, i)
tips = teaching_suggestions(agent.dataframe())
check("noise + dropping engagement produce tips", len(tips) >= 2)
recs = [{"student_id": "Student_1", "present_sec": 30, "attentive_sec": 5, "attention_pct": 16.7, "status": "Present"},
        {"student_id": "Student_2", "present_sec": 2, "attentive_sec": 2, "attention_pct": 100, "status": "Not marked"}]
sa = student_analytics(recs)
check("low-attention student flagged", "Low attention" in sa.loc[0, "remark"])
with tempfile.TemporaryDirectory() as d:
    s = save_reports(agent.dataframe(), sa, tips, outdir=d)
    check("reports written", all(os.path.exists(os.path.join(d, f)) for f in
                                 ["engagement_log.csv", "attendance.csv", "summary.json"]))

# ---- vision ----
tr = FaceTracker()
blank = np.zeros((480, 640, 3), dtype=np.uint8)
out = tr.update(blank, now=0.0)
out = tr.update(blank, now=1.0)
check("blank frame -> 0 faces, no crash", out["faces"] == 0 and tr.attendance_records() == [])
tr.close()

print(f"\nAll {passed} checks passed.")
