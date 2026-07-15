"""
=========================================================
Remote Blood Pressure Estimation using Webcam
=========================================================

Description
-----------
This application estimates systolic (SBP) and diastolic (DBP)
blood pressure from a webcam using remote photoplethysmography
(rPPG) signals extracted from the face and hand.


Controls
--------
Press 's' : Start a 30-second measurement
Press 'q' : Quit application
"""

# ======================================================
# Import required libraries
# ======================================================
#
# OpenCV      : webcam and image processing
# NumPy       : numerical operations
# Pandas      : saving measurements
# MediaPipe   : hand landmark detection
# SciPy       : filtering and peak detection
#
# ======================================================
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy import signal
import time
from collections import deque
from datetime import datetime


# ======================================================
# Face Detection
# ======================================================
#
# Haar Cascade classifier is used to detect the user's face.
# The detected face region is later used as the facial ROI
# for extracting remote PPG signals.
#
# ======================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ======================================================
# Hand Detection
# ======================================================
#
# MediaPipe Hands detects hand landmarks in every frame.
# The detected landmarks define the hand ROI from which
# the RGB signals are extracted.
#
# ======================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)


# ======================================================
# Subject Information
# ======================================================
#
# These anthropometric parameters are used by the
# vascular model for estimating blood pressure.
#
# ======================================================

AGE = 26
HEIGHT = 185.0
BMI = 33.6


# ======================================================
# Measurement Parameters
# ======================================================
#
# Defines:
# - measurement duration
# - update interval
# - signal freshness
# - physiological limits
# - buffer sizes
#
# ======================================================

MEASURE_DURATION_SEC = 30.0          
UPDATE_SEC = 2.0                     
FRESH_SEC = 1.5
CLEAR_HAND_AFTER = 6.0
CLEAR_FACE_AFTER = 6.0

BANDPASS_HZ = (0.7, 3.0)
MIN_PTT_SEC = 0.06
MAX_PTT_SEC = 0.80

MIN_WINDOW_SEC = 12.0
MAXLEN = 1200

SHOW_PTT = True


# ======================================================
# Automatic Calibration
# ======================================================
#
# During the first seconds of the measurement the program
# estimates a reference Pulse Transit Time.
#
# This removes the need for manual subject-specific
# calibration before each measurement.
#
# ======================================================

DEFAULT_SBP_MMHG = 120.0
DEFAULT_DBP_MMHG = 80.0

AUTO_REF_WARMUP_SEC = 12.0           
AUTO_REF_MIN_SAMPLES = 3
AUTO_REF_MAX_SAMPLES = 10
AUTO_REF_FALLBACK_PTT_SEC = 0.25

SBP_SLOPE = 0.205
DBP_SLOPE = 0.120

SBP_RANGE = (70.0, 200.0)
DBP_RANGE = (40.0, 130.0)


# ======================================================
# Signal Processing Functions
# ======================================================
#
# This section contains all functions required for
# preprocessing the RGB signals and estimating
# Pulse Transit Time.
#
# ======================================================

# ------------------------------------------------------
# POS Algorithm
# ------------------------------------------------------
#
# Applies the Plane-Orthogonal-to-Skin algorithm
# to convert RGB temporal signals into an rPPG signal.
#
# The method reduces motion artifacts and illumination
# variations while enhancing the pulsatile component.
#
# ------------------------------------------------------

def apply_pos(R, G, B, fps):
    R, G, B = map(lambda x: np.asarray(x, dtype=np.float64), (R, G, B))
    need = max(int(MIN_WINDOW_SEC * fps), 90)
    if len(R) < need:
        return None

    C = np.vstack((R, G, B))
    C -= C.mean(axis=1, keepdims=True)

    w = max(int(1.6 * fps), 10)
    step = max(w // 2, 1)

    M = np.array([[0, 1, -1],
                  [-2, 1,  1]], dtype=np.float64)

    S = np.zeros(len(R), dtype=np.float64)
    for i in range(0, len(R) - w + 1, step):
        H = M @ C[:, i:i + w]
        a = (np.std(H[0]) + 1e-8) / (np.std(H[1]) + 1e-8)
        S[i:i + w] += H[0] + a * H[1]

    S -= np.mean(S)
    sd = np.std(S)
    if sd > 1e-8:
        S /= sd
    return S

# ------------------------------------------------------
# Band-pass Filtering
# ------------------------------------------------------
#
# Filters the extracted pulse signal between
# 0.7 Hz and 3 Hz to keep only physiological
# heart-rate frequencies.
#
# ------------------------------------------------------

def bandpass(sig_in, fps, low_hz, high_hz, order=3):
    sig_in = np.asarray(sig_in, dtype=np.float64)
    if len(sig_in) < 10:
        return None
    b, a = signal.butter(order, [low_hz, high_hz], fs=fps, btype="band")
    return signal.filtfilt(b, a, sig_in)

# ------------------------------------------------------
# Pulse Transit Time Estimation
# ------------------------------------------------------
#
# Computes the delay between facial and hand pulse
# signals using peak matching.
#
# The median delay is considered the estimated
# Pulse Transit Time.
#
# ------------------------------------------------------

def repo_delay_seconds(sig_face, sig_hand, fps, invert=False):
    f = bandpass(sig_face, fps, *BANDPASS_HZ)
    h = bandpass(sig_hand, fps, *BANDPASS_HZ)
    if f is None or h is None:
        return None

    if invert:
        f = -f
        h = -h

    min_dist = int(0.35 * fps)
    prom_f = max(0.15, 0.35 * np.std(f))
    prom_h = max(0.15, 0.35 * np.std(h))

    vf, _ = signal.find_peaks(-f, distance=min_dist, prominence=prom_f)
    vh, _ = signal.find_peaks(-h, distance=min_dist, prominence=prom_h)

    if len(vf) < 3 or len(vh) < 3:
        return None

    lags = []
    j = 0
    for pf in vf:
        while j < len(vh) and vh[j] < pf:
            j += 1
        if j >= len(vh):
            break
        lag = (vh[j] - pf) / fps
        if MIN_PTT_SEC <= lag <= MAX_PTT_SEC:
            lags.append(lag)

    if len(lags) < 2:
        return None

    return float(np.median(lags))


# ======================================================
# Blood Pressure Estimation Model
# ======================================================
#
# The following functions model arterial properties
# and transform Pulse Transit Time into estimates
# of systolic and diastolic blood pressure.
#
# ======================================================

def vessel_length_from_height(height_cm, age):
    return (0.45 + 0.001 * age) * height_cm / 100.0


def vessel_radius(height_cm, age, bmi):
    return -0.258 + 0.029 * height_cm + 0.006 * age + 0.036 * bmi


def vessel_thickness(age, bmi):
    return 0.25 + 0.005 * age + 0.005 * bmi


def log_expression(ptt_sec):
    alpha = 0.017
    rho = 1060.0
    E0 = 1005.0

    L = vessel_length_from_height(HEIGHT, AGE)
    r = 0.001 * 0.5 * vessel_radius(HEIGHT, AGE, BMI)
    h = 0.001 * vessel_thickness(AGE, BMI)

    # safety floors
    L = max(float(L), 0.05)
    r = max(float(r), 0.0005)
    h = max(float(h), 0.0002)

    t = max(float(ptt_sec), 1e-4)
    return ((-2.0 / alpha) * np.log(t) +
            (1.0 / alpha) * np.log((2.0 * r * rho * (L ** 2)) / (h * E0)))


def _clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))


def make_bp_mapper(ref_ptt_sec, ref_sbp, ref_dbp):
    le_ref = log_expression(ref_ptt_sec)
    sbp_intercept = ref_sbp - SBP_SLOPE * le_ref
    dbp_intercept = ref_dbp - DBP_SLOPE * le_ref

    def sbp_from_ptt(ptt_sec):
        le = log_expression(ptt_sec)
        sbp = SBP_SLOPE * le + sbp_intercept
        return _clamp(sbp, *SBP_RANGE)

    def dbp_from_ptt(ptt_sec, sbp=None):
        le = log_expression(ptt_sec)
        dbp = DBP_SLOPE * le + dbp_intercept
        dbp = _clamp(dbp, *DBP_RANGE)
        if sbp is not None:
            dbp = min(dbp, float(sbp) - 5.0)
            dbp = max(dbp, DBP_RANGE[0])
        return float(dbp)

    return sbp_from_ptt, dbp_from_ptt


def timestamped_filename(prefix="bp_results", ext="csv"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


# ======================================================
# Buffers and Measurement State
# ======================================================
#
# Stores RGB signals, measurement history,
# smoothing buffers and session variables.
#
# ======================================================

# ROI traces
R_face = deque(maxlen=MAXLEN); G_face = deque(maxlen=MAXLEN); B_face = deque(maxlen=MAXLEN)
R_hand = deque(maxlen=MAXLEN); G_hand = deque(maxlen=MAXLEN); B_hand = deque(maxlen=MAXLEN)
T_buf = deque(maxlen=300)

# session logs
session_rows = []  # list of dict rows

# smoothing
sbp_smooth = deque(maxlen=5)
dbp_smooth = deque(maxlen=5)

# timing
last_update = 0.0
last_face_time = 0.0
last_hand_time = 0.0

# debug
last_ptt_sbp = None
last_ptt_dbp = None

# auto-ref
warmup_ptts = deque(maxlen=AUTO_REF_MAX_SAMPLES)
ref_ptt_sec = None
sbp_from_ptt, dbp_from_ptt = make_bp_mapper(AUTO_REF_FALLBACK_PTT_SEC, DEFAULT_SBP_MMHG, DEFAULT_DBP_MMHG)

# session state
IDLE = "IDLE"
MEASURING = "MEASURING"
state = IDLE
measure_start = None
session_start = None  


def reset_session_buffers():
    global session_rows, sbp_smooth, dbp_smooth
    global warmup_ptts, ref_ptt_sec, sbp_from_ptt, dbp_from_ptt
    global last_ptt_sbp, last_ptt_dbp, last_update
    global R_face, G_face, B_face, R_hand, G_hand, B_hand

    # keep camera running; reset measurement-specific stuff
    session_rows = []
    sbp_smooth.clear()
    dbp_smooth.clear()

    warmup_ptts.clear()
    ref_ptt_sec = None
    sbp_from_ptt, dbp_from_ptt = make_bp_mapper(AUTO_REF_FALLBACK_PTT_SEC, DEFAULT_SBP_MMHG, DEFAULT_DBP_MMHG)

    last_ptt_sbp = None
    last_ptt_dbp = None
    last_update = time.time()

    # reset traces so each session is clean
    R_face.clear(); G_face.clear(); B_face.clear()
    R_hand.clear(); G_hand.clear(); B_hand.clear()


def save_session_csv(rows):
    if not rows:
        print("Session ended: no values to save.")
        return None
    df = pd.DataFrame(rows)
    fname = timestamped_filename()
    df.to_csv(fname, index=False, float_format="%.6f")
    print(f"Saved {fname} ({len(df)} rows)")
    return fname


# ======================================================
# Main Processing Loop
# ======================================================
#
# The following loop continuously:
#
# 1. Captures webcam frames
# 2. Detects face and hand
# 3. Extracts ROIs
# 4. Updates RGB signals
# 5. Computes PTT
# 6. Estimates blood pressure
# 7. Displays results
#
# ======================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("Controls: press 's' to start 30s measurement, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    T_buf.append(now)

    # FPS estimate
    if len(T_buf) > 30:
        dt = np.diff(np.array(T_buf))
        dt = dt[(dt > 0) & (dt < 1.0)]
        fps = float(1.0 / np.mean(dt)) if len(dt) else 30.0
    else:
        fps = 30.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------
    # Face ROI Extraction
    #
    # Detect the face and calculate the average RGB
    # values inside the facial region.
    # ------------------------------------------------------

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        last_face_time = now
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = frame[y:y + h, x:x + w]
        if roi.size > 0:
            R_face.append(float(roi[:, :, 2].mean()))
            G_face.append(float(roi[:, :, 1].mean()))
            B_face.append(float(roi[:, :, 0].mean()))

    # ------------------------------------------------------
    # Hand ROI Extraction
    #
    # Detect the hand landmarks and calculate the
    # average RGB values inside the hand region.
    # ------------------------------------------------------
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        last_hand_time = now
        lm = res.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        H, W, _ = frame.shape
        xs = [int(p.x * W) for p in lm.landmark]
        ys = [int(p.y * H) for p in lm.landmark]
        x1, x2 = max(min(xs), 0), min(max(xs), W)
        y1, y2 = max(min(ys), 0), min(max(ys), H)
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            R_hand.append(float(roi[:, :, 2].mean()))
            G_hand.append(float(roi[:, :, 1].mean()))
            B_hand.append(float(roi[:, :, 0].mean()))
    else:
        if (now - last_hand_time) > CLEAR_HAND_AFTER:
            R_hand.clear(); G_hand.clear(); B_hand.clear()
            last_ptt_sbp = None
            last_ptt_dbp = None

    # Clear face if stale
    if (now - last_face_time) > CLEAR_FACE_AFTER:
        R_face.clear(); G_face.clear(); B_face.clear()
        last_ptt_sbp = None
        last_ptt_dbp = None

    have_face = (now - last_face_time) < FRESH_SEC
    have_hand = (now - last_hand_time) < FRESH_SEC

    # -------------------------
    # MEASUREMENT STATE MACHINE
    # -------------------------
    if state == MEASURING:
        elapsed = now - measure_start
        remaining = max(0.0, MEASURE_DURATION_SEC - elapsed)

        # stop condition
        if elapsed >= MEASURE_DURATION_SEC:
            state = IDLE
            saved = save_session_csv(session_rows)
            reset_session_buffers()  # prep for next session
        else:
            # periodic compute only while measuring
            if (now - last_update) >= UPDATE_SEC:
                SBP = DBP = None

                if have_face and have_hand:
                    n = min(len(R_face), len(R_hand))
                    need = max(int(MIN_WINDOW_SEC * fps), 90)

                    if n >= need:
                        f_sig = apply_pos(list(R_face)[-n:], list(G_face)[-n:], list(B_face)[-n:], fps)
                        h_sig = apply_pos(list(R_hand)[-n:], list(G_hand)[-n:], list(B_hand)[-n:], fps)

                        if f_sig is not None and h_sig is not None:
                            ptt_sbp = repo_delay_seconds(f_sig, h_sig, fps, invert=False)
                            ptt_dbp = repo_delay_seconds(f_sig, h_sig, fps, invert=True)

                            if ptt_sbp is not None and ptt_dbp is not None:
                                last_ptt_sbp = float(ptt_sbp)
                                last_ptt_dbp = float(ptt_dbp)

                                # Auto-ref collection within the session
                                if ref_ptt_sec is None:
                                    if (now - session_start) <= AUTO_REF_WARMUP_SEC:
                                        warmup_ptts.append(last_ptt_sbp)
                                        if len(warmup_ptts) >= AUTO_REF_MIN_SAMPLES:
                                            ref_ptt_sec = float(np.median(np.array(warmup_ptts)))
                                            sbp_from_ptt, dbp_from_ptt = make_bp_mapper(
                                                ref_ptt_sec, DEFAULT_SBP_MMHG, DEFAULT_DBP_MMHG
                                            )
                                    else:
                                        ref_ptt_sec = AUTO_REF_FALLBACK_PTT_SEC
                                        sbp_from_ptt, dbp_from_ptt = make_bp_mapper(
                                            ref_ptt_sec, DEFAULT_SBP_MMHG, DEFAULT_DBP_MMHG
                                        )

                                SBP = float(sbp_from_ptt(last_ptt_sbp))
                                DBP = float(dbp_from_ptt(last_ptt_dbp, sbp=SBP))

                                # smooth
                                sbp_smooth.append(SBP)
                                dbp_smooth.append(DBP)
                                SBP = float(np.median(np.array(sbp_smooth)))
                                DBP = float(np.median(np.array(dbp_smooth)))

                                session_rows.append({
                                    "unix_time": now,
                                    "time_iso": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
                                    "SBP": SBP,
                                    "DBP": DBP,
                                    "PTT_S": last_ptt_sbp,
                                    "PTT_D": last_ptt_dbp,
                                    "fps": fps,
                                    "ref_ptt_sec": ref_ptt_sec if ref_ptt_sec is not None else np.nan
                                })

                last_update = now

    # ------------------------------------------------------
    # Graphical User Interface
    #
    # Displays:
    # - measurement status
    # - remaining time
    # - estimated SBP
    # - estimated DBP
    # - Pulse Transit Time
    #
    # ------------------------------------------------------
    if state == IDLE:
        cv2.putText(frame, "IDLE - press 's' to start 30s measurement", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        elapsed = now - measure_start
        remaining = max(0.0, MEASURE_DURATION_SEC - elapsed)
        cv2.putText(frame, f"MEASURING... {remaining:0.1f}s left", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if session_rows:
            last = session_rows[-1]
            cv2.putText(frame, f"SBP: {last['SBP']:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"DBP: {last['DBP']:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if SHOW_PTT and last_ptt_sbp is not None:
                cv2.putText(frame, f"PTT(S): {last_ptt_sbp:.3f}s", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if ref_ptt_sec is None:
                cv2.putText(frame, "Auto-ref: warming up...", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"REF_PTT(auto): {ref_ptt_sec:.3f}s", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Collecting signal...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # status hints
    if not (have_face and have_hand) and state == MEASURING:
        cv2.putText(frame, "Need FACE + HAND in view", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Blood Pressure Estimation", frame)

    # ------------------------------------------------------
    # Keyboard Controls
    #
    # S : Start measurement
    # Q : Quit application
    #
    # ------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if key == ord("s") and state == IDLE:
        reset_session_buffers()
        state = MEASURING
        measure_start = time.time()
        session_start = measure_start
        last_update = 0.0  # so it tries to compute quickly
        print("Measurement started (30s).")


cap.release()
cv2.destroyAllWindows()
