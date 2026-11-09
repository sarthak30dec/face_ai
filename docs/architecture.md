# Architecture

```
            image / video frame (BGR uint8)
                        │
                 ┌──────▼───────┐   FaceDetection(bbox, score, 5 keypoints, track_id?)
                 │  detection   │   yunet (default) | scrfd | torchvision (fine-tuned)
                 └──────┬───────┘
          ┌─────────────┼───────────────────────────┐
          │             │                           │
 ┌────────▼───────┐ ┌───▼─────────────────┐ ┌───────▼─────────┐
 │ recognition/   │ │ landmarks/          │ │ detection/      │
 │ alignment      │ │ landmarker          │ │ tracking (video)│
 │ (5-pt → 112²)  │ │ crop → MediaPipe    │ │ temporary ids   │
 └────────┬───────┘ │ 478-pt mesh         │ └───────┬─────────┘
 ┌────────▼───────┐ └───┬─────────────────┘         │
 │ embedder       │     │ FaceLandmarks             │
 │ sface|arcface  │     │                           │
 └────────┬───────┘     │                           │
 ┌────────▼───────┐     │                           │
 │ matcher +      │     │                           │
 │ gallery        │     │                           │
 │ (cosine, open  │     │                           │
 │  set, Unknown) │     │                           │
 └────────┬───────┘     │                           │
          │ IdentityMatch                           │
          └─────────────┼───────────────────────────┘
                 ┌──────▼───────┐
                 │visualization │  render_overlay(image, detections, landmarks, identities)
                 └──────────────┘
```

## Layers

| Layer | Package | Depends on | Notes |
|---|---|---|---|
| Typed results | `face_ai.api.schemas` | numpy | `BoundingBox`, `FaceDetection`, `FaceLandmarks`, `FaceEmbedding`, `IdentityMatch`, `FaceAnalysis`; each has `to_dict()` |
| Components | `detection`, `recognition`, `landmarks`, `visualization` | schemas, OpenCV, ONNX Runtime, MediaPipe | Each backend sits behind a small protocol (`FaceDetector`, `FaceEmbedder`) |
| Service | `face_ai.api.service.FaceAIService` | components + YAML configs | Lazy model loading; the stable interface below |
| Data/eval tooling | `face_ai.data`, `face_ai.evaluation` | — | Not needed at inference time |
| CLIs | `scripts/*.py` | service / tooling | Thin wrappers; no business logic |

The core never imports the CLIs, and nothing below `api.service` reads command-line state, so the same code serves the
demo, the evaluation scripts and a future app.

## Stable interface

| Function | Input | Output |
|---|---|---|
| `detect_faces(image)` | BGR uint8 H×W×3 | `list[FaceDetection]`, sorted by score |
| `extract_embedding(aligned_face)` | 112×112 BGR crop from `align()` | `FaceEmbedding` (L2-normalized; `model` names the embedder) |
| `enroll_person(person_id, images, consent)` | ids, paths/arrays, `ConsentRecord(granted=True, …)` | `EnrollmentReport` (accepted count, rejections with reasons) |
| `identify_face(image, face, gallery=None)` | image + `FaceDetection` | `IdentityMatch` (`label` is the id or `"Unknown"`, `similarity`, `threshold`, top-k) |
| `extract_landmarks(image, face)` | image + `FaceDetection` | `FaceLandmarks` (478×3) or `None` |
| `render_overlay(image, detections, landmarks, identities, options)` | aligned lists | annotated copy |
| `FaceAIService.analyze(image)` | image | `list[FaceAnalysis]`, the full pipeline |

Module-level versions of the first five exist in `face_ai.api.service` and use a default service built from
`configs/`.

## Design decisions

* **Two scores, never mixed.** `FaceDetection.score` means "this is a face". `IdentityMatch.similarity` means "this
  face looks like person X". The overlay labels them `det` and `sim`, and the threshold for each is configured separately.
* **Open set by construction.** `Matcher.identify` returns `person_id=None` (label `Unknown`) unless the best cosine
  similarity reaches a threshold calibrated on held-out identities at a target false-positive identification rate.
* **One alignment for all embedders.** Every embedder uses the 5-point ArcFace template (Umeyama similarity
  transform). This matches OpenCV's `FaceRecognizerSF.alignCrop`: our SFace embeddings equal OpenCV's to cosine
  0.9999999.
* **Mesh from the same face.** `extract_landmarks` crops around *our* detection (square crop, 1.8× box, resized to
  256 px) and runs MediaPipe on the crop. Mesh, box and identity therefore always refer to the same face, including
  in crowds where MediaPipe's own short-range detector would miss small faces.
* **Galleries are data, not models.** Adding a person adds embeddings; nothing is retrained. A gallery records which
  embedder produced it and refuses to load with a different one. Evaluation galleries are `ephemeral` and cannot be saved.
* **Licensing-aware defaults.** The default stack (YuNet MIT, SFace Apache-2.0, MediaPipe Apache-2.0) is permissive.
  SCRFD/ArcFace (non-commercial) are opt-in via config for research or comparison.
* **CPU ONNX Runtime by default.** It is deterministic and fast enough for these models. `providers: coreml` is available.
