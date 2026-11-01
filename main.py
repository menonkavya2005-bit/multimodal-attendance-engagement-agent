"""Milestone 6 (Activity 6.1): run the core application.

Examples:
    python main.py                      # webcam + microphone
    python main.py --no-audio           # webcam only
    python main.py --video class.mp4 --audio-file class.wav
Press  q  in the video window to stop and generate reports.
"""
import argparse
import time
import cv2

from vision.vision_module import FaceTracker
from audio.audio_module import AudioMonitor, AudioFileSource
from agent.multimodal_agent import MultimodalAgent
from analytics.analytics import teaching_suggestions, student_analytics, save_reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--video", help="video file instead of webcam")
    ap.add_argument("--audio-file", help="audio file instead of microphone")
    ap.add_argument("--no-audio", action="store_true", help="disable audio")
    ap.add_argument("--max-seconds", type=float, default=0, help="auto-stop (0 = until q)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("Could not open the video source. Check the camera / file path.")
        return

    audio = None
    if not args.no_audio:
        audio = AudioFileSource(args.audio_file) if args.audio_file else AudioMonitor()
        if not audio.start():
            audio = None

    tracker, agent = FaceTracker(), MultimodalAgent()
    t0, last_log, latest = time.time(), -1, None
    print("Running... press 'q' in the video window to stop.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # video files use the video clock so results do not depend on CPU speed
        elapsed = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) if args.video else (time.time() - t0)
        vision = tracker.update(frame, now=elapsed)
        sound = audio.latest(elapsed) if audio else {"state": "unavailable", "rms": 0.0}

        if int(elapsed) != last_log:  # one fused reading per second
            last_log = int(elapsed)
            latest = agent.step(vision, sound, elapsed)

        tracker.draw(frame)
        if latest:
            txt = f"Engagement {latest['engagement']:.0f} ({latest['level']}) | audio: {latest['audio_state']}"
            cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Attendance & Engagement Agent", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if args.max_seconds and elapsed >= args.max_seconds:
            break

    cap.release()
    cv2.destroyAllWindows()
    if audio:
        audio.stop()
    tracker.close()

    log_df = agent.dataframe()
    students = student_analytics(tracker.attendance_records())
    tips = teaching_suggestions(log_df)
    summary = save_reports(log_df, students, tips)

    print("\n===== SESSION SUMMARY =====")
    print(f"Average engagement : {summary['avg_engagement']}")
    print(f"Students present   : {summary['students_present']} / {summary['students_detected']} detected")
    print(students.to_string(index=False) if len(students) else "No students detected.")
    print("\nSuggestions:")
    for t in tips:
        print(" -", t)
    print("\nReports saved in the data/ folder. Launch dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()
