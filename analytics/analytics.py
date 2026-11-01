"""Milestone 5: teaching suggestions and student analytics."""
import json
import os
import pandas as pd
from config import LOW_ATTENTION_STUDENT_PCT, HIGH_ENGAGEMENT, DATA_DIR


# ---------- Activity 5.1: teaching suggestions ----------
def teaching_suggestions(log_df):
    if log_df is None or log_df.empty:
        return ["Not enough data collected yet. Run a longer session."]
    active = log_df[log_df["faces"] > 0]
    if active.empty:
        return ["No participants were detected. Check camera position and lighting."]

    tips = []
    avg = active["engagement"].mean()
    half = len(active) // 2
    trend = 0.0
    if half >= 3:
        trend = active["engagement"].iloc[half:].mean() - active["engagement"].iloc[:half].mean()
    noise_frac = (active["audio_state"] == "noise").mean()
    speech_frac = (active["audio_state"] == "speech").mean()
    silence_frac = (active["audio_state"] == "silence").mean()
    attention_frac = active["attentive"].sum() / max(active["faces"].sum(), 1)

    if noise_frac > 0.30:
        tips.append("High background noise detected. Reduce noise or ask students to mute / quieten down.")
    if attention_frac < 0.50:
        tips.append("Many students are looking away. Try a short activity, poll or question to regain focus.")
    if trend < -10:
        tips.append("Engagement is dropping over the session. Consider a short break or a quick recap.")
    elif trend > 10:
        tips.append("Engagement is rising over the session. Keep the current teaching style.")
    if speech_frac > 0.60 and avg < HIGH_ENGAGEMENT:
        tips.append("Mostly one-way talking with moderate engagement. Add interactive questions or a discussion.")
    if silence_frac > 0.70:
        tips.append("Long silent periods. Ask questions to encourage participation.")
    if not tips:
        tips.append("Engagement looks healthy. Continue with the current approach.")
    return tips


# ---------- Activity 5.2: student analytics ----------
def student_analytics(records):
    cols = ["student_id", "present_sec", "attentive_sec", "attention_pct", "status", "remark"]
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records)

    def remark(r):
        if r["status"] != "Present":
            return "Seen too briefly to mark attendance"
        if r["attention_pct"] < LOW_ATTENTION_STUDENT_PCT:
            return "Low attention - may need support"
        return "Engaged"

    df["remark"] = df.apply(remark, axis=1)
    return df[cols]


def save_reports(log_df, students_df, tips, outdir=DATA_DIR):
    os.makedirs(outdir, exist_ok=True)
    log_df.to_csv(os.path.join(outdir, "engagement_log.csv"), index=False)
    students_df.to_csv(os.path.join(outdir, "attendance.csv"), index=False)
    active = log_df[log_df["faces"] > 0] if not log_df.empty else log_df
    summary = {
        "avg_engagement": round(float(active["engagement"].mean()), 1) if len(active) else 0.0,
        "students_present": int((students_df["status"] == "Present").sum()) if len(students_df) else 0,
        "students_detected": int(len(students_df)),
        "duration_sec": float(log_df["elapsed_sec"].max()) if len(log_df) else 0.0,
        "suggestions": tips,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
