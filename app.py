"""Milestone 6 (Activity 6.2): optional Streamlit dashboard.  Run: streamlit run app.py"""
import json
import os
import pandas as pd
import streamlit as st
from config import DATA_DIR

st.set_page_config(page_title="Attendance & Engagement Agent", layout="wide")
st.title("Multimodal Attendance & Engagement Agent")

log_p = os.path.join(DATA_DIR, "engagement_log.csv")
att_p = os.path.join(DATA_DIR, "attendance.csv")
sum_p = os.path.join(DATA_DIR, "summary.json")

if not (os.path.exists(log_p) and os.path.exists(sum_p)):
    st.info("No session data yet. Run `python main.py` first, then refresh this page.")
    st.stop()

log = pd.read_csv(log_p)
students = pd.read_csv(att_p) if os.path.exists(att_p) else pd.DataFrame()
with open(sum_p) as f:
    summary = json.load(f)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg engagement", summary["avg_engagement"])
c2.metric("Students present", f"{summary['students_present']} / {summary['students_detected']}")
c3.metric("Session length (s)", int(summary["duration_sec"]))
dominant = log["audio_state"].mode()[0] if len(log) else "n/a"
c4.metric("Dominant audio", dominant)

st.subheader("Engagement over time")
if len(log):
    st.line_chart(log.set_index("elapsed_sec")["engagement"])

st.subheader("Audio activity")
if len(log):
    st.bar_chart(log["audio_state"].value_counts())

st.subheader("Student attendance & analytics")
if len(students):
    st.dataframe(students, use_container_width=True)
else:
    st.write("No students detected in this session.")

st.subheader("Teaching suggestions")
for tip in summary["suggestions"]:
    st.write("•", tip)
