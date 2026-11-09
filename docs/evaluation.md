# Evaluation

All numbers below were produced on this machine on 2026-10-30 by the scripts named in each section. Raw JSON is in
`outputs/metrics/`, plots in `outputs/visualizations/`, and a compiled table in `outputs/reports/summary.md`
(`scripts/make_report.py`).

**Hardware:** Apple M4 Pro (12 CPU cores, 16-core GPU), 24 GB RAM, macOS 26.7.1, Python 3.11.15, ONNX Runtime 1.31.1
(CPU provider, default threads), OpenCV 5.0.0, MediaPipe 1.2.0, PyTorch 2.15.0 (MPS for training/inference of the
torchvision model). Timings were taken with no other heavy process running. Each latency excludes disk I/O, uses batch
size 1, and starts after 10 untimed warm-up calls.

---

## 1. Detection: WIDER FACE val (3,226 images, 39,708 faces)

**Protocol:** a Python port of the official `eval_tools` (2017-03-31): IoU ≥ 0.5, global min-max score normalization,
1,000 thresholds, faces outside each setting's keep list ignored, VOC all-point AP. Detectors run at score ≥ 0.02 so the
PR curve is complete. WIDER val is our **test** split: it was never used for training, checkpoint selection or
threshold tuning.

**Port validation:** our pipeline reproduces the published APs of both pretrained detectors to within about 0.01, so
the port and the decoding are trustworthy:

| Detector | Ours easy / medium / hard | Published easy / medium / hard |
|---|---|---|
| YuNet 2023mar | **0.888 / 0.872 / 0.770** | 0.884 / 0.871 / 0.761 (OpenCV Zoo) |
| SCRFD-10GF (640²) | **0.954 / 0.940 / 0.827** | 0.952 / 0.939 / 0.831 (InsightFace) |
| Faster R-CNN MBv3, fine-tuned run 1 | 0.809 / 0.684 / 0.343 | — |
| Faster R-CNN MBv3, fine-tuned run 2 | 0.872 / 0.807 / 0.480 | — |

![PR curves](../outputs/visualizations/wider_pr_all.png)

### Operating point (raw score threshold; "hard" keep list = 31,958 faces)

A *false detection* is a box that matches no face in the keep list (IoU < 0.5, or a duplicate). Boxes on unlabeled
faces and boxes with loose localization also count, so this is an upper bound on true false alarms. A *missed face* is a
kept face with no matching box.

| Detector | thr | Precision | Recall | False det. | Missed | Recall <16 px | 16–32 px | 32–64 px | ≥64 px |
|---|---|---|---|---|---|---|---|---|---|
| YuNet | 0.6 | 0.873 | 0.689 | 3,201 | 9,936 | 0.484 | 0.753 | 0.866 | 0.908 |
| SCRFD | 0.5 | 0.969 | 0.629 | 644 | 11,855 | 0.293 | 0.753 | 0.903 | 0.961 |
| FT run 2 | 0.5 | 0.585 | 0.462 | 10,463 | 17,203 | 0.092 | 0.515 | 0.840 | 0.934 |

At these default thresholds SCRFD trades recall for precision. Its curve dominates YuNet's at every recall level (see
the AP figures), so lowering SCRFD's threshold gives more recall than YuNet at equal precision. Most misses are faces
under 32 px, which matter little for recognition: the enrollment minimum is 64 px, and identity is unreliable below about 40 px.

### Detector performance (full WIDER val pass; median image 1024×754, max 1024×5541)

| Detector | p50 ms | p95 ms | images/s | model load s | model size | peak process RSS |
|---|---|---|---|---|---|---|
| YuNet (CPU, OpenCV) | 8.7 | 16.2 | 98.9 | 0.01 | 0.22 MB | 1.2 GB* |
| SCRFD (CPU, ORT, 640² letterbox) | 36.6 | 47.8 | 27.0 | 0.04 | 16.1 MB | 1.4 GB* |
| FT Faster R-CNN (MPS) | 42.5 | 63.0 | 21.4 | 2.2 | 144.5 MB | 2.8 GB* |

\*Peak RSS of the whole evaluation process (Python + NumPy + all loaded libraries + ground truth), not of the model alone.

---

## 2. Detector fine-tuning experiment

**Why this experiment:** YuNet's own training stack (libfacedetection.train, mmdet-based) is heavy to reproduce,
and training a detector from scratch is not justified. So we ran **transfer learning** of a COCO-pretrained
torchvision `fasterrcnn_mobilenet_v3_large_fpn`: new 2-class head, anchors re-sized to 8–128 px, horizontal-flip
augmentation, SGD with momentum, warm-up plus cosine LR, gradient clipping, seed 42. It trains on `train_ft` and selects on `val_ft`
(300 images; simple all-face VOC AP at IoU 0.5). Targets: WIDER `invalid`/degenerate boxes and boxes < 8 px are
excluded (counted below). Checkpoints `last.pt`/`best.pt` include optimizer, scheduler and RNG state, and
`--resume` was verified (it restarted at the saved epoch/step).

| Run | Train images | Input scale | Epochs | Steps | Wall time (MPS) | val_ft AP per epoch | WIDER val AP (easy/med/hard) |
|---|---|---|---|---|---|---|---|
| 1 | 3,000 (28,867 boxes; 494 flagged, 3,838 < 8 px skipped) | short side 640 | 3 | 2,250 | 11.9 min | 0.269 → 0.306 → 0.331 | 0.809 / 0.684 / 0.343 |
| 2 | 6,000 (59,600 boxes; 1,146 flagged, 9,857 < 8 px skipped) | short side 800 | 2 | 3,000 | 23.4 min | 0.455 → 0.490 | 0.872 / 0.807 / 0.480 |

Artifacts: `models/checkpoints/detector_ft_run1/`, `models/checkpoints/detector_ft/` (run 2), each with `history.json`,
`config.json`, `train.log`; summaries in `outputs/metrics/detector_training{,_run1}.json`.

**Conclusion:** fine-tuning works and scales as expected: more data and higher resolution give +0.06 / +0.12 / +0.14 AP.
But after ~35 minutes of laptop training it remains clearly **below** YuNet, especially on small faces. It is also
5× slower and 650× larger. **YuNet stays the default detector.** Closing the gap would need full-dataset training for
tens of epochs, multi-scale augmentation and a detector designed for small faces. That is a multi-day GPU job and was
not attempted. The run is reported as an experiment, not as an improvement.

---

## 3. Recognition: LFW

Alignment for every embedder uses YuNet keypoints (`data/processed/lfw/faces_yunet.json`). YuNet found the central
face in all 13,233 images, so there were 0 fallbacks.

### 3a. Verification: View 2, 6,000 pairs, 10 folds

10-fold accuracy picks each fold's threshold on the other nine folds. ROC, AUC, EER and TAR@FAR are computed over all
6,000 pairs. Thresholds for those are chosen on the same pairs, so they describe the ROC rather than giving held-out estimates.

| Embedder | 10-fold accuracy | AUC | EER | TAR @ FAR 1e-3 | TAR @ FAR 1e-2 | TAR @ FAR 1e-1 | mean genuine / impostor cos |
|---|---|---|---|---|---|---|---|
| SFace | **99.35 % ± 0.37** | 0.99877 | 0.82 % | 98.80 % | 99.27 % | 99.67 % | 0.654 / 0.084 |
| ArcFace R50 | **99.85 % ± 0.17** | 0.99931 | 0.18 % | 99.73 % | 99.83 % | 99.87 % | 0.661 / 0.004 |

Published LFW: SFace 99.40 %, ArcFace R50 (WebFace600K) ≈ 99.8 %. Our full pipeline (our detector → our alignment →
model) reproduces both.

FAR 1e-3 corresponds to 3 of the 3,000 impostor pairs, so it is a coarse estimate.

**FAR / FRR at the deployed open-set threshold** (§3b): SFace @ 0.464: FAR 0.00 %, FRR 5.83 %. ArcFace @ 0.298:
FAR 0.00 %, FRR 0.50 %. The identification threshold is deliberately strict for 1:N use. For 1:1 verification, a
threshold near the 10-fold optimum (SFace ≈ 0.35–0.40) gives a better FAR/FRR balance.

![SFace ROC and DIR/FPIR](../outputs/visualizations/recognition_sface.png)

### 3b. Open-set identification (identity-disjoint protocol, `data/splits/lfw/openset.json`)

The gallery has 2 enrollment images per known identity; known probes are other images of those people; unknown probes
are people who are not enrolled. **The threshold is chosen on the calibration identities only** (460 enrolled /
1,674 known / 2,414 unknown probes) to hit the target FPIR. It is then applied unchanged to the disjoint **test**
identities (441 enrolled / 1,656 known / 2,434 unknown probes). Matching uses the mean template per person.

| Embedder | Target FPIR | Threshold (calib.) | Test rank-1 (closed set) | Test DIR | Test FPIR | Unknown rejection | Known rejected as Unknown | Mis-identification |
|---|---|---|---|---|---|---|---|---|
| SFace | 1 % | **0.464** | 99.58 % | 98.67 % | 0.99 % | 99.01 % | 1.33 % | 0.00 % |
| SFace | 5 % | 0.424 | 99.58 % | 99.21 % | 4.44 % | 95.56 % | 0.72 % | 0.06 % |
| ArcFace R50 | 1 % | **0.298** | 99.88 % | 99.82 % | 0.78 % | 99.22 % | 0.18 % | 0.00 % |
| ArcFace R50 | 5 % | 0.259 | 99.88 % | 99.88 % | 4.23 % | 95.77 % | 0.12 % | 0.00 % |

The calibrated FPIR transfers to unseen identities (1 % target → 0.99 % / 0.78 % on test), so the threshold
generalizes on this data. The bold thresholds are written to `configs/recognition.yaml`. Max-over-references matching
gives nearly identical results (`identification.max` in the JSON). Below FPIR ≈ 1e-3 the DIR/FPIR curve rests on one
or two unknown probes, so it is not estimable with this data.

**These numbers are not a promise of performance in your setting.** LFW is mostly frontal, well-lit photos of public
figures, and its 2-image galleries are cleaner than real enrollment photos. Recalibrate on consented, in-domain
data, and check error rates per demographic group, before relying on a threshold.

### 3c. Embedder performance (112×112 aligned input; excludes detection/alignment)

| Embedder | p50 ms/face | p95 ms/face | faces/s | model load s | ONNX size | dim |
|---|---|---|---|---|---|---|
| SFace | 3.84 | 3.99 | 260 | 0.05 | 36.9 MB (18.4 MB Core ML FP16) | 128 |
| ArcFace R50 | 17.04 | 17.48 | 58 | 0.06 | 166.3 MB | 512 |

Embedding the 11,075 LFW images used by the two protocols took about 1 min (SFace) and about 4 min (ArcFace) on CPU,
including JPEG decode and alignment.

---

## 4. End-to-end pipeline (`scripts/benchmark_pipeline.py`)

The image is a 1024×681 group photo with 13 faces; 200 timed runs per stage, 10 warm-up.

| Stage | p50 | p95 | Load time |
|---|---|---|---|
| YuNet detect (whole image) | 7.5 ms | 8.7 ms | < 0.01 s |
| SCRFD detect (whole image) | 35.2 ms | 40.6 ms | 0.01 s |
| SFace embed, per face | 3.8 ms | 4.0 ms | 0.02 s |
| ArcFace embed, per face | 17.1 ms | 17.5 ms | 0.05 s |
| MediaPipe mesh, per face (crop mode) | 3.5 ms | 4.0 ms | 0.78 s |
| Overlay rendering (13 faces, no mesh) | 4.7 ms | — | — |
| **Full `analyze()` with the default stack (13 faces: detect + 13 × align/embed/identify + 13 × mesh)** | **103.7 ms** | **105.8 ms** | — |

Peak RSS of the benchmark process with all models loaded: about 1.0 GB. A typical single-face selfie frame costs
roughly 7.5 + 3.8 + 3.5 ≈ 15 ms (~65 FPS) on CPU. The 60-frame 640×480 video demo ran at 18 FPS with about 7 faces per
frame, mesh included.

## 5. What is *not* measured

* On-device iOS latency/accuracy (Core ML parity is checked on macOS only; see `docs/ios_integration.md`).
* Demographic differentials (LFW cannot support this).
* Video identity stability over time, anti-spoofing (photos/screens/masks), and image quality assessment.
* Recognition with real enrollment photos of consenting users. No such gallery exists in this project.

---

## 6. Recognition accuracy enhancements (2026-10-30)

Goal: decide reliably whether a face is a given enrolled person, and say "uncertain" instead of guessing.
All numbers come from `scripts/evaluate_recognition.py` on the same LFW protocols as §3. Calibration uses the
calibration identities only; results are on the disjoint test identities.

### 6a. Pipeline variants measured

| Variant | LFW 10-fold | EER | TAR @ FAR 1e-3 | Test DIR @ FPIR 1 % | ms / face |
|---|---|---|---|---|---|
| ArcFace R50, YuNet keypoints (**default**) | 99.85 % ± 0.17 | 0.18 % | 99.73 % | 99.82 % | 16.6 |
| ArcFace R50 + flip, YuNet keypoints | 99.83 % ± 0.20 | 0.20 % | 99.73 % | 99.82 % | ~33 |
| ArcFace R50 + flip, SCRFD keypoints | 99.78 % ± 0.18 | 0.33 % | 99.70 % | 99.82 % | ~33 + refiner |
| SFace, YuNet keypoints (old default) | 99.35 % ± 0.37 | 0.82 % | 98.80 % | 98.67 % | 3.8 |
| **SFace + flip, SCRFD keypoints** | **99.60 % ± 0.20** | **0.48 %** | **99.23 %** | **99.40 %** | 8.0 + refiner |

* **ArcFace:** flip test-time augmentation and refined keypoints change nothing measurable on LFW (the differences
  are 1–4 of 6,000 pairs), while doubling the cost, so the default ArcFace pipeline uses neither.
* **SFace** clearly benefits from both together (EER −41 %, missed known people 1.3 % → 0.6 %), so the SFace profile
  enables them. The SFace-with-flip-only variant was not run, so the two effects are not separated.
* LFW is close to saturated for ArcFace; harder data (pose, age, low resolution) is needed to separate these variants
  better.

### 6b. Calibrated decision thresholds (`configs/calibration/*.json`)

Impostor statistics come from **1,878,806** stranger-vs-template comparisons and 1,674 genuine comparisons
(calibration half). That resolves false-match rates down to ~5e-7.

| Embedder | 1:N accept (FPIR 1 %) | 1:1 verify accept (FAR 1e-4) | Test 1:1 FAR | Test 1:1 FRR |
|---|---|---|---|---|
| ArcFace R50 | 0.298 | 0.265 | 0.97e-4 (1.8 M impostor pairs) | 0.12 % |
| SFace + flip + SCRFD | 0.472 | 0.442 | 0.86e-4 | 0.42 % |

The calibrated FAR transfers to unseen identities: target 1e-4, measured 0.97e-4 and 0.86e-4. The "reject" threshold
(1 % of genuine scores) came out at or above the accept threshold for both models, so on LFW there is no
score-based uncertain band. Uncertainty comes from face quality and the margin rule.

### 6c. Full decision rule on test identities (ArcFace, default config)

| | correct match | wrong person | missed (Unknown) | uncertain |
|---|---|---|---|---|
| Enrolled people (1,656 probes), no quality gate | 99.82 % | 0.00 % | 0.18 % | — |
| Enrolled people, quality gate + margin 0.05 | 97.71 % | 0.00 % | 0.12 % | 2.17 % |

| | false match | Unknown | uncertain |
|---|---|---|---|
| Strangers (2,434 probes), no quality gate | 0.78 % | 99.22 % | — |
| Strangers, quality gate + margin 0.05 | 0.78 % | 97.08 % | 2.14 % |

**Does the quality gate help?** It flags about 3 % of LFW faces. Those faces are riskier: in 1:1 verification their
false-reject rate is 2.1 % vs 0.06 % for acceptable faces, and LFW pair errors are 2× (ArcFace) and 4.7× (SFace)
higher when a low-quality face is involved. On LFW the gate converts those risky cases into "uncertain". It does
**not** reduce the 0.78 % stranger false matches, which come from good-quality look-alikes and LFW label noise; only
the accept threshold controls those. LFW is mostly good-quality photos, so the gate should matter more on live camera
frames, but that has not been measured. The margin rule (0.05) never triggered on LFW; it is a safeguard for
galleries that contain look-alikes such as siblings.

**Reading the numbers:** a stranger gets matched to *someone* in a 441-person gallery 0.78 % of the time. In a 1:1
check against one named person, the measured false-accept rate is about 1 in 10,000. For "is this person X?", use
verify mode in the UI or `verify_face()`.

### 6d. UI check (2026-10-30, at the user's request)

The UI's own endpoints were driven with a temporary gallery of 3 LFW public figures (3, 3 and 1 reference photos),
deleted afterwards. The user's `gallery/` was not touched.

* **Enrollment:** refused without consent or with an invalid id. Rejected a head-turned photo (quality 0.41), an
  impostor photo slipped into a set (cos 0.01 to the others), a group photo, a 17 px face, and blurred photos.
* **Identify, 1:N:** 298/300 held-out photos of enrolled people correct, **0 wrong person**, 2 uncertain.
  400 strangers: 393 Unknown, 7 uncertain, **0 false matches**.
* **Verify, 1:1, against one person:** 99/100 genuine photos "same person" (1 uncertain); 200 strangers: 196
  "different person", 4 uncertain, **0 false accepts**.
* **Score separation:** genuine 5th percentile 0.55 (median 0.75); highest stranger score 0.21.
* **Fixed during this check:** enrollment crashed (a module-level `enroll_person` shadowed the library function);
  the page scrolled sideways at phone width; verify mode labelled faces with the 1:N result and counted "recognized"
  from 1:N; the "stranger chance" was shown for Unknown results.
* **Observed:** even mild blur (Gaussian kernel 9) makes a face "Uncertain" although ArcFace still scored the right
  person at 0.69. The sharpness gate may be stricter than necessary; tune it on real camera photos.

### 6e. Live video in the UI (synthetic clip, 2026-10-30)

The UI's "Live video" tab was driven in the browser pane with a 16 s, 1280×720, 15 fps H.264 clip built from held-out
LFW photos (Bush and Powell enrolled in a temporary gallery, deleted afterwards). The clip had four segments: Bush
moving with a stranger beside him; Powell zooming with 1 s of heavy blur; all three side by side with Bush's photo
swapped mid-segment; a pan across a 13-person crowd of strangers. Frames are sent at 960 px.

| | Track averaging on (default) | Frame by frame |
|---|---|---|
| Frames analysed | 207 | 202 |
| Correct names for Bush / Powell | every frame they were visible | every sharp frame |
| Wrong-person matches / false matches | **0** | **0** |
| Powell during the 1 s blur | still **identified** (15-frame history) | **Uncertain** every frame (quality 0–0.07) |
| Crowd strangers | 129 Unknown, 22 Uncertain | 150 Unknown, 29 Uncertain |

Throughput: about 15 frames/s with one face and about 11.5 frames/s with three (detect 6 ms, ArcFace about 25 ms per
face, mesh about 4 ms per face, plus the browser round trip). Track ids stayed stable through movement, zoom and the
photo swap.

The camera path (`getUserMedia`) was not exercised: no camera was used. Still not measured: real camera footage and
galleries with many enrolled people. False matches grow with gallery size.

### 6f. Tilted and turned faces (2026-10-30)

40 frontal LFW faces were rotated in-plane; "verifies" means 1:1 similarity to the same face upright is at or above
the calibrated accept threshold (0.265).

| Tilt | Face detected | Mesh placed (mean error, % of eye distance) | Verifies: detector keypoints only | Verifies: with mesh cross-check (default) |
|---|---|---|---|---|
| 0–30° | 40/40 | 0.8–1.4 % | 39–40/40 | 40/40 |
| 45° | 38/40 | 2.2 % | 28/38 | **38/38** |
| 60° | 37/40 | 2.1 % | 10/37 | **37/37** |
| 75° | 28/40 | 1.5 % | 0/28 | **28/28** |
| 90° | 26/40 | 3.2 % | 0/26 | **23/26** |

Past ~45° of tilt, YuNet keeps returning an upright-looking but misplaced set of 5 keypoints (28 px off at 60°), so
recognition failed while the mesh stayed accurate. `FaceAIService.check_keypoints` now compares the detector
keypoints with 5 points derived from the mesh. If the mesh shows more than 20° of tilt, or the two disagree by more
than 35 % of the eye distance, the face crop is rotated upright, re-detected and aligned. This touches 0.9 % of
ordinary LFW photos (19/2000), so the calibrated thresholds still apply. It costs one mesh pass per face (about 3.5 ms).
The remaining gap at 75–90° is the detector missing sideways faces.

Head turn (yaw), LFW photos of the same person against one frontal reference:

| Yaw (nose offset / eye distance) | n | Mesh found | 1:1 verifies |
|---|---|---|---|
| < 0.15 (frontal) | 300 | 100 % | 99.7 % |
| 0.15–0.30 (slight) | 1,138 | 99.8 % | 99.9 % |
| 0.30–0.45 (about half turn) | 168 | 100 % | 100 % |
| > 0.45 (strong) | 43 | 90.7 % | 97.7 % |

LFW contains no full profiles, so faces turned beyond about 60–70° are not measured. The quality gate
(`yaw_bad: 0.55`) marks strongly turned faces "Uncertain" in identification, by design.
