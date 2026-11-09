# CLAUDE.md

Local face detection, recognition and face-mesh module (Python 3.11), built to be ported to iOS later.

## Rules for Claude

- **Git:** never `git commit`, `git push`, open PRs or touch any remote unless the user explicitly asks in that
  conversation. The project is a private GitHub repo (`sarthak30dec/face_ai`); never commit anything `.gitignore` excludes.
- **No self-testing:** do not write or run automated tests (pytest etc.) and do not take screenshots or drive the
  browser/UI to check your own work. The user tests. You may run the project's data and evaluation scripts when a
  change needs them, for example to recalibrate thresholds after changing the recognition pipeline, and must report
  their numbers honestly.
- **Privacy:** never enroll people from public datasets (`data/raw/`), never upload images, embeddings or galleries
  anywhere, and never log identity names above DEBUG. `gallery/` holds private biometric data: don't read, copy or
  print its contents.
- **Profile cards** (name, occupation, about, link in `gallery.json`) are typed by the user.
- **Licensing:** SCRFD and ArcFace (`models/pretrained/buffalo_l/`) are non-commercial research only. YuNet (MIT),
  SFace (Apache-2.0) and MediaPipe (Apache-2.0) are permissive. State the license impact whenever a default changes.
- Do not claim accuracy you have not measured. Record measured numbers in `docs/evaluation.md`.

## Commands (run from this folder)

```bash
./run.sh                                             # browser UI (sets up .venv + models on first run)
.venv/bin/python scripts/run_demo.py --input IMG_OR_VIDEO [--gallery gallery/] [--no-mesh]
.venv/bin/python scripts/manage_gallery.py enroll|list|remove|purge ...
.venv/bin/python scripts/evaluate_recognition.py [--update-config]   # LFW verification + open-set calibration
.venv/bin/python scripts/evaluate_detector.py        # WIDER FACE val, official protocol
.venv/bin/python scripts/train_detector.py [--resume models/checkpoints/detector_ft/last.pt]
.venv/bin/python scripts/make_report.py              # outputs/reports/summary.md
```

Setup from scratch: `download_datasets.py` → `download_models.py` → `validate_datasets.py` → `preprocess_datasets.py`.

## Layout

- `src/face_ai/api/`: `schemas.py` (typed results), `service.py` (stable interface: `detect_faces`,
  `extract_embedding`, `enroll_person`, `identify_face`, `verify_face`, `extract_landmarks`, `render_overlay`),
  `web_ui.py` + `static/index.html` (local UI, stdlib HTTP server bound to 127.0.0.1).
- `src/face_ai/detection/`: YuNet / SCRFD / fine-tuned torchvision detectors, IoU tracker, training.
- `src/face_ai/recognition/`: alignment (5-point ArcFace template + optional SCRFD keypoint refiner), embedders
  (flip test-time augmentation), quality assessment, calibration, gallery, matcher (match / unknown / uncertain
  decision, 1:1 verify), enrollment (quality + consistency checks), temporal aggregation for video.
- `src/face_ai/landmarks/`: MediaPipe Face Landmarker (visualization only, never used for identity).
- `src/face_ai/evaluation/`: WIDER protocol port, verification/identification metrics, benchmarks.
- `configs/*.yaml`: all thresholds and model paths. Recognition thresholds are calibrated values; change them only
  via `evaluate_recognition.py --update-config` or with measured justification.
- `configs/calibration/<embedder>_<flip>_<align>.json`: thresholds + score distributions from LFW calibration
  identities, used for decisions and false-match estimates (scores only, no biometrics). Regenerate with
  `evaluate_recognition.py --update-config` after any change to alignment, embedder or flip settings.
- Large data and artifacts (`data/raw`, `data/processed`, `models/*`, `gallery/`, `outputs/predictions`) are git-ignored.

## Conventions

- Images are BGR `uint8` numpy arrays; coordinates are pixels in the original image.
- Detection score ("is this a face") and identity similarity ("is this person X") are separate and must never be mixed.
- Keep the core library independent of the CLI scripts and UI; scripts stay thin.
- Match the existing style: type hints, dataclasses for results, short docstrings, configs in YAML.
