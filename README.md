# face_ai

A local, privacy-first Python module for **face detection**, **face recognition** (open-set, with `Unknown`), and a
**wireframe face mesh** overlay, with automated dataset download/validation and reproducible evaluation. It is designed
to be ported to iOS later (see `docs/ios_integration.md`).

| Component | Default model | License | Measured (see `docs/evaluation.md`) |
|---|---|---|---|
| Detection | YuNet 2023mar | MIT | WIDER val AP 0.888 / 0.872 / 0.770 (easy/med/hard) |
| Recognition | ArcFace-R50 (512-d) | **non-commercial research only** | LFW 99.85 %; 1:1 FAR 1e-4 @ FRR 0.12 %; 0 wrong-person matches on test ids |
| Recognition (permissive alternative) | SFace + flip + SCRFD keypoints | Apache-2.0 (SCRFD refiner: research only) | LFW 99.60 %; 1:1 FAR 1e-4 @ FRR 0.42 % |
| Mesh | MediaPipe Face Landmarker (478 pts) | Apache-2.0 | visualization only |
| Optional detector | SCRFD-10GF | non-commercial | AP 0.954 / 0.940 / 0.827 |

Every identification ends in one of three decisions: **match**, **Unknown** or **Uncertain**. Uncertain covers low
face quality, two enrolled people too alike, or a score between thresholds, and is never given a name. Use
`verify_face()` (or verify mode in the UI) to ask "is this person X?" with a calibrated 1-in-10,000 false-accept rate.
Switching `backend` in `configs/recognition.yaml` requires re-enrolling people.

## Run the UI

```bash
./run.sh
```

That's the only command you need. The first run creates `.venv`, installs the dependencies and downloads the models,
so it needs internet and takes a few minutes. Later runs start straight away. It opens `http://localhost:8000`, or the
next free port if that one is busy. Press Ctrl+C to stop.

You need Python 3.11 or [uv](https://docs.astral.sh/uv/). The datasets aren't required to run the UI. The download
includes ArcFace-R50, the default recognizer, which is **non-commercial research only**.

To use another port, or to start without opening a browser:

```bash
PORT=9000 ./run.sh
```

```bash
./run.sh --no-open
```

In the page:

- **Photo:** upload, drag and drop, or paste an image. You see boxes, the face mesh with its shard mask, and a table of
  detection score, identity and similarity per face. Switch detectors and adjust both thresholds live.
- **Live feed:** runs the same pipeline on your webcam or a video file. The mask and brackets are computed in the page
  on every video frame, so they stay on the face, while identity comes from the Python server a few times a second.
  Each face is identified from the quality-weighted average of its recent frames, so a blurred or turned frame doesn't
  flip the name.
- **Click any face** to open its dossier. Register an Unknown face with a name, occupation, short note and optional
  profile link (consent confirmation required). Matched faces then show that card in the HUD. You can edit a matched
  face's card or add the current frame as an extra reference.

Sound effects play on loading, identity results, clicks and the dossier, and the **Sound** toggle mutes them. Browsers
keep audio off until your first click or key press, so the boot sounds may be silent on a first load. The samples are
CC0 from [Kenney](https://kenney.nl/assets/category:Audio) (`src/face_ai/api/static/sfx/`).

The page binds to 127.0.0.1 only and uses no web fonts or CDNs. Uploaded images are processed in memory and never
written to disk. Only enrollments are saved, to `gallery/`. Cards are typed by you, and nothing is looked up
automatically. Animations turn off when the OS "reduce motion" setting is on.

## Full setup (datasets, evaluation, training)

Only needed for the steps below, not for the UI.

```bash
cd face_ai
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[train,export,dev]"
```

```bash
.venv/bin/python scripts/download_datasets.py     # WIDER FACE (train/val/annotations/eval tools) + LFW, verified
```

```bash
.venv/bin/python scripts/download_models.py       # YuNet, SFace, buffalo_l (SCRFD+ArcFace), MediaPipe
```

```bash
.venv/bin/python scripts/validate_datasets.py     # integrity, duplicates, box sanity -> outputs/reports/dataset_validation.md
```

```bash
.venv/bin/python scripts/preprocess_datasets.py --lfw-detectors yunet scrfd   # COCO conversion, splits, LFW face cache
```

All steps are idempotent: valid downloads are reused, interrupted downloads resume, and checksums are re-verified.

## Evaluation

```bash
.venv/bin/python scripts/evaluate_detector.py                       # YuNet + SCRFD on WIDER val (official protocol)
```

```bash
.venv/bin/python scripts/evaluate_recognition.py --update-config    # LFW verification + open-set id; writes calibrated thresholds
```

```bash
.venv/bin/python scripts/evaluate_detector.py --detectors torchvision --checkpoint models/checkpoints/detector_ft/best.pt
```

Results: `outputs/metrics/*.json`, plots in `outputs/visualizations/`.

## Fine-tuning the detector (optional)

```bash
.venv/bin/python scripts/train_detector.py                          # settings in configs/detection.yaml -> training
```

```bash
.venv/bin/python scripts/train_detector.py --resume models/checkpoints/detector_ft/last.pt
```

Checkpoints (`last.pt`, `best.pt`), `history.json`, `config.json` and `train.log` are written to `models/checkpoints/detector_ft/`.

## Demo

```bash
.venv/bin/python scripts/run_demo.py --input path/to/photo.jpg --output outputs/visualizations/photo_annotated.jpg
```

```bash
.venv/bin/python scripts/run_demo.py --input clip.mp4 --gallery gallery/ --det-threshold 0.6   # identity thresholds come from configs/calibration
```

Options: `--no-mesh`, `--mesh-groups face_oval lips …`, `--keypoints`, `--json`, `--detector scrfd`,
`--embedder arcface_r50`, `--max-frames N`, `--display`. Each face shows its box, the mesh, `det` (detector
confidence), the identity or `Unknown`, `sim` (cosine similarity to the best enrolled person), and in video a
temporary `track #n`. Track ids are not identities.

## Enrolling people (with consent)

```bash
.venv/bin/python scripts/manage_gallery.py enroll --person-id alice --images ~/consented/alice/ --consent-granted-by "Alice" --purpose "door unlock demo" --retention-days 365
```

```bash
.venv/bin/python scripts/manage_gallery.py list
```

```bash
.venv/bin/python scripts/manage_gallery.py remove --person-id alice
```

Adding a person never retrains a model; it stores 1–N reference embeddings in `gallery/` (git-ignored, 0600 files).
Enrollment refuses images from `data/raw/` and requires a consent record.

## Python API

```python
from face_ai.api.service import FaceAIService
from face_ai.recognition.gallery import ConsentRecord
from face_ai.utils.image_io import load_image

svc = FaceAIService.from_configs()
img = load_image("group.jpg")
faces = svc.detect_faces(img)                                   # list[FaceDetection]
emb = svc.extract_embedding(svc.align(img, faces[0]))           # FaceEmbedding (L2-normalized)
svc.enroll_person("alice", ["a1.jpg", "a2.jpg"], ConsentRecord(True, "Alice", "demo"))
match = svc.identify_face(img, faces[0])                        # IdentityMatch: .decision match/unknown/uncertain
check = svc.verify_face(img, faces[0], "alice")                 # VerificationResult: is this alice? (1:1)
mesh = svc.extract_landmarks(img, faces[0])                     # FaceLandmarks (478 x 3) or None
vis = svc.render_overlay(img, faces, [mesh], [match])
results = svc.analyze(img)                                      # the whole pipeline: list[FaceAnalysis]
```

## Layout

See `docs/architecture.md`. Datasets: `docs/datasets.md`. Models and licenses: `docs/model_cards.md`. Metrics and
protocols: `docs/evaluation.md`. Porting: `docs/ios_integration.md`.

## Privacy

Everything runs locally; nothing is uploaded. Galleries store embeddings plus consent metadata and image hashes, never
images or paths. Identity names are not logged. To delete one person, use `manage_gallery.py remove`; to delete the
whole gallery, `manage_gallery.py purge`. For production, keep `FACE_AI_GALLERY_DIR` on an encrypted volume, enforce
`retention_days`, and keep the gallery out of backups and Git (already ignored).

## Tests

Automated tests are left to the project owner (`tests/unit`, `tests/integration`, `tests/fixtures` are empty
placeholders). Integration tests can rely on `data/raw/` and `models/pretrained/` being present after the commands above.
