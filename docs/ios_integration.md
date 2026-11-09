# Future iOS integration

This phase builds no app. This document records what was verified on this machine and what the Swift port has to
reproduce.

## 1. Component mapping

| Python component | iOS option (recommended first) | Verified here | Notes |
|---|---|---|---|
| YuNet detector | Core ML `yunet.mlpackage` (ML Program, FP16, iOS 16+) | ✓ converted ONNX → PyTorch (onnx2torch) → Core ML; raw head outputs within 0.056 abs of ONNX Runtime on 8 real images | Decoding + NMS (≈ 60 lines) must be ported from OpenCV's `FaceDetectorYN`. Input fixed at 640×640; letterbox in Swift or convert with `ct.RangeDim` |
| (alternative detector) | Apple Vision `VNDetectFaceLandmarksRequest` | — | Zero model weight and on-device optimized, but returns its own landmark set; still needs the 5 alignment points mapped to the ArcFace template |
| SFace embedder | Core ML `sface.mlpackage` (18.4 MB FP16) | ✓ cosine ≥ 0.999998 vs ONNX Runtime on 8 real aligned faces | Input 1×3×112×112, RGB, raw 0–255 floats |
| Alignment | `CGAffineTransform` / vImage / Core Image warp with the same Umeyama solution | — | Use the same 5-point template (`ARCFACE_TEMPLATE`). Alignment mismatch is the most common cause of accuracy loss in ports |
| Face mesh | MediaPipe Tasks iOS SDK (`MediaPipeTasksVision`, `FaceLandmarker`) with the same `face_landmarker.task` | — | Topology lists are identical (`FaceLandmarksConnections`) |
| Gallery | On-device file encrypted with a key in the Secure Enclave / Keychain; Data Protection class `complete` | — | Same schema as `gallery.json` + embeddings; never sync to iCloud without explicit consent |
| ONNX alternative | ONNX Runtime Mobile (iOS) with the CoreML execution provider | — | Runs the exact ONNX files in `models/exports/onnx/`; larger binary |

`python scripts/export_models.py` regenerates `models/exports/{onnx,coreml}/` and `outputs/metrics/export_report.json`.
ArcFace-R50 also converts (`--models arcface_r50`), but its license is non-commercial, so it is excluded by default.
The fine-tuned torchvision Faster R-CNN is **not** a good on-device candidate (two-stage, RoI ops, large).

## 2. Contracts the Swift code must match

* **Image:** BGR in Python. In Swift you will usually have RGB(A) `CVPixelBuffer`s, so swap channels exactly as noted in
  `models/exports/onnx/*.json` (YuNet expects BGR; SFace expects RGB).
* **Keypoint order:** image-left eye, image-right eye, nose, image-left mouth corner, image-right mouth corner.
* **Embedding:** L2-normalize before comparing; similarity = dot product.
* **Thresholds:** the values in `configs/recognition.yaml` were calibrated with Python/ONNX Runtime FP32 and
  YuNet keypoints. FP16 Core ML changes scores slightly (≤ 6e-3 abs here). **Re-run calibration through the iOS
  pipeline** (or at least on its embeddings) before relying on a threshold, and recalibrate on in-domain, consented
  data from your target cameras.
* **Result objects:** mirror `face_ai/api/schemas.py` as Swift structs (`BoundingBox`, `FaceDetection`,
  `FaceLandmarks`, `IdentityMatch`, `FaceAnalysis`). Keep `detection.score` and `identity.similarity` as separate fields.

## 3. Suggested Swift surface

```swift
protocol FaceAI {
    func detectFaces(_ image: CVPixelBuffer) throws -> [FaceDetection]
    func extractEmbedding(_ alignedFace: CVPixelBuffer) throws -> FaceEmbedding
    func enrollPerson(_ personID: String, images: [CVPixelBuffer], consent: ConsentRecord) throws -> EnrollmentReport
    func identifyFace(_ image: CVPixelBuffer, face: FaceDetection, gallery: Gallery) throws -> IdentityMatch
    func extractLandmarks(_ image: CVPixelBuffer, face: FaceDetection) throws -> FaceLandmarks?
    func renderOverlay(_ image: CVPixelBuffer, detections: [FaceDetection],
                       landmarks: [FaceLandmarks?], identities: [IdentityMatch?]) -> CGImage
}
```

## 4. Parity test plan for the port

1. Export 50 LFW images plus their Python outputs (boxes, keypoints, aligned crops, embeddings) as golden fixtures.
2. On device, assert: box IoU > 0.95, keypoints within 1.5 px, aligned-crop PSNR > 35 dB, embedding cosine > 0.999.
3. Re-run the open-set calibration (`scripts/evaluate_recognition.py` logic) on embeddings produced by the iOS pipeline.
4. Measure on-device latency (warm-up 10, then p50/p95 over ≥ 200 frames) for each model on the oldest supported device.

## 5. Privacy on device

Process locally and never upload frames, crops or embeddings. Show and store the consent record at enrollment.
Support per-person deletion (`Gallery.remove`) and full purge (`Gallery.delete`). Apply the retention period in
`ConsentRecord.retention_days`. Exclude the gallery from backups unless the user opts in.
