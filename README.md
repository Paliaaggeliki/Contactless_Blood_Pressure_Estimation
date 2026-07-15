# Contactless_Blood_Pressure_Estimation

This application estimates Systolic (SBP) and Diastolic (DBP) blood pressure using a standard webcam through **remote Photoplethysmography (rPPG)** and the **Pulse Transit Time (PTT)** method. 
Developed as part of my internship, this project offers a non-contact physiological measurement method by extracting rPPG signals from both the user's face and hand to calculate the pulse propagation delay.

## Code Architecture & Logical Breakdown

The source code in `camera4.py` is structured into 5 distinct logical sections. Here is how each part functions:

### 1. Face & Hand Detection
* **Face Detection (Haar Cascade):** Uses OpenCV's `CascadeClassifier` to locate and track the user's face in real-time.
* **Hand Detection (MediaPipe):** Uses Google's MediaPipe Hands framework to detect hand landmarks and track keypoints.

### 2. Region of Interest (ROI) Extraction
* Once the face and hand bounding areas are detected, the system crops these regions (ROIs).
* It calculates the average RGB color values for each frame and appends them to temporal buffers (`R_face`, `G_face`, `R_hand`, etc.) to construct the raw signals.
* If a region is lost (e.g., the hand leaves the frame), the corresponding buffers are automatically cleared after a timeout.

### 3. Signal Preprocessing (rPPG Extraction)
* **POS Algorithm (Plane-Orthogonal-to-Skin):** The `apply_pos()` function converts raw RGB signals into a clean rPPG signal. This mathematical method minimizes motion artifacts and illumination changes.
* **Band-pass Filtering:** The `bandpass()` function uses a Butterworth filter to isolate frequencies between $0.7 \text{ Hz}$ and $3.0 \text{ Hz}$. This targets the natural human heart rate range (approx. 42 to 180 BPM).

### 4. Blood Pressure Estimation Model
* **Pulse Transit Time (PTT):** Calculated in `repo_delay_seconds()` by measuring the phase delay between the facial pulse peaks and hand pulse peaks.
* **Hemodynamic Formulas:** Using the user's physical parameters (Age, Height, BMI) and vascular modeling, the PTT is mathematically converted to SBP and DBP estimates.
* **Auto-Calibration:** During the first 12 seconds of a measurement session, the application automatically establishes a baseline reference PTT. This removes the need for manual, user-specific calibration.

### 5. Main Loop & Graphical User Interface (GUI)
* The central `while` loop captures camera frames, coordinates the detection frameworks, executes the pipelines, and displays real-time SBP/DBP overlays.
* Pressing **'s'** triggers a 30-second recording. Once complete, all calculated parameters are automatically exported to a timestamped `.csv` file.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/Paliaaggeliki/Contactless_Blood-Pressure-Estimation.git](https://github.com/Paliaaggeliki/Contactless_Blood-Pressure-Estimation.git)
cd Contactless_Blood-Pressure-Estimation
