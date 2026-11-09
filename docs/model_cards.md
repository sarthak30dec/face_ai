# Model cards

All weights are downloaded by `scripts/download_models.py` into `models/pretrained/`. SHA-256 values are in
`configs/models.yaml` and `data/manifests/model_*.json`. The metrics below are **our measurements** (see
`docs/evaluation.md`), not vendor claims, unless marked "published".

## Selection summary

| Role | Default (permissive) | Reference / optional | Why the default |
|---|---|---|---|
| Detection | **YuNet 2023mar** (MIT) | SCRFD-10GF (research-only), fine-tuned Faster R-CNN MobileNetV3 (ours) | 0.23 MB, ~9 ms per WIDER image on CPU, 5 keypoints for alignment, converts to Core ML; SCRFD is more accurate (+6.6 AP easy) but non-commercial and ~5× slower |
| Recognition | **ArcFace R50 / WebFace600K** (research-only), changed 2026-10-30 for accuracy | SFace + flip + SCRFD keypoints (Apache-2.0 model) | Most accurate measured: LFW 99.85 %, 1:1 FAR 1e-4 at 0.12 % FRR. **Its license is non-commercial:** for any release, switch `backend: sface` (99.60 % with flip + refined keypoints) or license the weights |
| Landmarks | **MediaPipe Face Landmarker** (Apache-2.0) | — | Official 478-point topology, native iOS SDK, 3.6 MB |

Alternatives considered and not used: RetinaFace (weights on Google Drive, superseded by SCRFD from the same group);
YOLO-face variants (AGPL-3.0, which is incompatible with an app that is not open source); dlib ResNet (99.38 % LFW, but
C++ build and no Core ML path); FaceNet/InceptionResnetV1 (larger and older than SFace with similar accuracy);
AdaFace/TransFace (weights trained on MS1M/WebFace, research licenses, larger). Apple Vision face detection is noted
in `docs/ios_integration.md` as an iOS-native alternative.

---

## YuNet (face_detection_yunet_2023mar.onnx)

* **Source:** OpenCV Zoo, `opencv/face_detection_yunet` (Hugging Face mirror owned by OpenCV); SHA-256 verified.
* **License:** MIT (model and code). Trained on WIDER FACE, whose license is CC BY-NC-ND. Whether dataset terms carry
  over to trained weights is legally unsettled; have commercial use reviewed.
* **Architecture:** anchor-free, depthwise-separable CNN; 75 k parameters; outputs boxes, 5 keypoints, score.
* **Our WIDER FACE val AP (official protocol):** easy 0.888 / medium 0.872 / hard 0.770 (published: 0.884 / 0.871 / 0.761).
* **Known weaknesses:** faces < 16 px (≈ 30 % recall), heavy occlusion, extreme pose. Keypoints are less precise on
  profiles, which hurts alignment.

## SCRFD-10GF (buffalo_l/det_10g.onnx)

* **Source:** InsightFace `buffalo_l` release v0.7 (GitHub release asset). Hash recorded on first download and pinned.
* **License:** InsightFace pretrained models are for **non-commercial research only**.
* **Our WIDER FACE val AP (640×640 letterbox, single scale):** easy 0.954 / medium 0.940 / hard 0.827 (published: 0.952 / 0.939 / 0.831).
* **Use:** accuracy reference and optional backend (`backend: scrfd`).

## SFace (face_recognition_sface_2021dec.onnx)

* **Source:** OpenCV Zoo, `opencv/face_recognition_sface`; SHA-256 verified. MobileFaceNet backbone trained with the
  sigmoid-constrained hypersphere loss (Zhong et al., 2021).
* **License:** Apache-2.0 (OpenCV Zoo). OpenCV Zoo does not document which face dataset these weights were trained on.
  The SFace paper trains on public web-face datasets (CASIA-WebFace / VGGFace2 / MS1M variants), some of which were
  later withdrawn. Confirm provenance before commercial use.
* **Output:** 128-d embedding. Input: 112×112 crop aligned to the ArcFace template, RGB, raw 0–255.
* **Our results:** LFW 10-fold accuracy 99.35 % ± 0.37 (published 99.40 %), AUC 0.9988, TAR 98.8 % @ FAR 1e-3.
  Open-set (test identities, threshold 0.464 calibrated at FPIR 1 %): rank-1 99.6 %, DIR 98.7 %, FPIR 0.99 %.

## ArcFace ResNet-50 / WebFace600K (buffalo_l/w600k_r50.onnx)

* **Source:** InsightFace `buffalo_l`. **License: non-commercial research only.** Training data: WebFace600K, a
  subset of WebFace260M, which itself requires a signed agreement.
* **Output:** 512-d. Input: aligned 112×112, (x − 127.5) / 127.5, RGB.
* **Our results:** LFW 99.85 % ± 0.17, AUC 0.9993, TAR 99.7 % @ FAR 1e-3. Open-set test (threshold 0.298): rank-1
  99.9 %, DIR 99.8 %, FPIR 0.78 %.

## MediaPipe Face Landmarker (face_landmarker.task, float16 v1)

* **Live feed:** the same model also runs in the page through `tasks_vision` (`@mediapipe/tasks-vision` 1.1.1,
  Apache-2.0), served locally with no CDN, so the mask is computed on the frame on screen. Video frames stay in the page
  for this step.

* **Source:** Google `storage.googleapis.com/mediapipe-models`, as linked from the official docs. Hash recorded on
  first download and pinned. **License:** Apache-2.0. Google's model card describes the training data and a fairness
  evaluation across geographic regions and skin tones.
* **Contents:** BlazeFace short-range detector + 478-point mesh (468 + 10 iris) + 52 blendshapes (unused here).
* **Use here:** wireframe visualization of the face oval, eyes, eyebrows, nose and lips using the official connection
  lists. **It is not used for identity.**
* **Known weakness:** designed for faces that fill much of the frame. We therefore run it on a crop around each
  detection (1.8× box, 256 px), which also makes it work in group photos.

## Fine-tuned Faster R-CNN MobileNetV3-FPN (ours)

* **Results (WIDER val, official protocol):** run 1 (3k images, 3 epochs, 12 min) AP 0.809 / 0.684 / 0.343;
  run 2 (6k images, 800 px, 2 epochs, 23 min) AP 0.872 / 0.807 / 0.480. Both are below YuNet (0.888 / 0.872 / 0.770),
  and the model is 144.5 MB and ~43 ms/image on MPS. Details: `docs/evaluation.md` §2.
* **Initialization:** torchvision COCO weights (`fasterrcnn_mobilenet_v3_large_fpn`, BSD-3 code; COCO images under
  Flickr terms). New 2-class box head; anchors resized to 8–128 px with face aspect ratios.
* **Training data:** WIDER FACE `train_ft` subset; selection on `val_ft`; never on the official val/test.
* **License status:** derived from COCO-pretrained weights plus WIDER (non-commercial), so research use only.
* **Status:** experimental; not the default detector (see the evaluation for why).

## Responsible-use notes (all recognition models)

* Error rates differ across demographic groups. LFW is not balanced and cannot measure this. Before any deployment,
  measure FAR/FRR per group on consented, representative data.
* A match is a probabilistic similarity, not proof of identity. Do not use it as the only factor for decisions with
  legal or safety consequences.
* Embeddings are biometric data. Store them encrypted, honor deletion, and never upload them.
