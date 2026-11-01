"""Milestone 2: face detection, tracking and presence-based attendance."""
import time
import cv2
import mediapipe as mp
from config import (MIN_DETECTION_CONF, PRESENCE_THRESHOLD_SEC, LOST_TIMEOUT_SEC,
                    MATCH_DIST_FRAC, ATTENTION_YAW_RATIO)


class Track:
    """One tracked face = one anonymous student."""

    def __init__(self, tid, center, now):
        self.id = tid
        self.center = center
        self.first_seen = now
        self.last_seen = now
        self.present_sec = 0.0
        self.attentive_sec = 0.0
        self.attentive = False
        self.visible = True
        self.bbox = None


class FaceTracker:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=MIN_DETECTION_CONF)
        self.tracks = {}
        self.next_id = 1
        self._last_t = None

    # ---------- Activity 2.1: detection + tracking ----------
    @staticmethod
    def _is_attentive(kp):
        """Face looks toward the camera if the nose sits between the eyes."""
        eye_r, eye_l, nose = kp[0], kp[1], kp[2]
        eye_dist = abs(eye_r.x - eye_l.x)
        if eye_dist < 1e-6:
            return False
        mid = (eye_r.x + eye_l.x) / 2
        return abs(nose.x - mid) / eye_dist < ATTENTION_YAW_RATIO

    def update(self, frame_bgr, now=None):
        now = time.time() if now is None else now
        dt = 0.0 if self._last_t is None else max(0.0, min(now - self._last_t, 1.0))
        self._last_t = now

        h, w = frame_bgr.shape[:2]
        result = self.detector.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        dets = []
        for d in (result.detections or []):
            bb = d.location_data.relative_bounding_box
            x, y = int(bb.xmin * w), int(bb.ymin * h)
            bw, bh = int(bb.width * w), int(bb.height * h)
            dets.append(((x + bw // 2, y + bh // 2), (x, y, bw, bh),
                         self._is_attentive(d.location_data.relative_keypoints)))

        for t in self.tracks.values():
            t.visible = False
            t.attentive = False

        active = [t for t in self.tracks.values() if now - t.last_seen <= LOST_TIMEOUT_SEC]
        max_d = MATCH_DIST_FRAC * w
        pairs = []
        for t in active:
            for di, (c, _, _) in enumerate(dets):
                dist = ((t.center[0] - c[0]) ** 2 + (t.center[1] - c[1]) ** 2) ** 0.5
                if dist <= max_d:
                    pairs.append((dist, t.id, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            self._mark_seen(self.tracks[tid], dets[di], now, dt)

        for di, det in enumerate(dets):  # unmatched face = new student
            if di not in used_d:
                t = Track(self.next_id, det[0], now)
                self.tracks[t.id] = t
                self.next_id += 1
                self._mark_seen(t, det, now, 0.0)

        visible = [t for t in self.tracks.values() if t.visible]
        return {"faces": len(visible), "attentive": sum(t.attentive for t in visible)}

    @staticmethod
    def _mark_seen(t, det, now, dt):
        t.center, t.bbox, t.attentive = det[0], det[1], det[2]
        t.visible = True
        t.last_seen = now
        t.present_sec += dt
        if det[2]:
            t.attentive_sec += dt

    # ---------- Activity 2.2: presence-based attendance ----------
    def attendance_records(self):
        rows = []
        for t in self.tracks.values():
            pct = 100 * t.attentive_sec / t.present_sec if t.present_sec > 0 else 0.0
            rows.append({
                "student_id": f"Student_{t.id}",
                "present_sec": round(t.present_sec, 1),
                "attentive_sec": round(t.attentive_sec, 1),
                "attention_pct": round(pct, 1),
                "status": "Present" if t.present_sec >= PRESENCE_THRESHOLD_SEC else "Not marked",
            })
        return rows

    def draw(self, frame):
        for t in self.tracks.values():
            if not t.visible or t.bbox is None:
                continue
            x, y, bw, bh = t.bbox
            color = (0, 200, 0) if t.attentive else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            tag = f"S{t.id} {t.present_sec:.0f}s" + (" OK" if t.present_sec >= PRESENCE_THRESHOLD_SEC else "")
            cv2.putText(frame, tag, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    def close(self):
        self.detector.close()
