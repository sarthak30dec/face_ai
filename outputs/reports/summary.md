# face_ai experiment summary

Hardware: `{"os": "Darwin 25.6.0", "os_version": "26.7.1", "machine": "arm64", "python": "3.11.15", "cpu": "Apple M4 Pro", "cpu_count_logical": 12, "ram_gb": 24.0, "disk_free_gb": 24.5, "torch": "2.15.0", "mps_available": true, "cuda_available": false, "onnxruntime": "1.31.1", "onnxruntime_providers": ["CoreMLExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"], "opencv": "5.0.0"}`

## Datasets

| dataset | download | validation | archive bytes | counts |
|---|---|---|---|---|
| wider_face | downloaded | valid_with_warnings | 1840392962 | {'train': 12880, 'val': 3226, 'test': 0} |
| lfw | downloaded | valid | 180804660 | {'images': 13233, 'identities': 5749} |

## Detection - WIDER FACE val (official protocol, IoU 0.5)

| detector | AP easy | AP medium | AP hard | thr | P (hard) | R (hard) | false det. | missed | R <16px | R >=64px | p50 ms | p95 ms | img/s | load s | size MB | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scrfd | 0.9535 | 0.9395 | 0.8269 | 0.5 | 0.969 | 0.629 | 644 | 11855 | 0.293 | 0.961 | 36.6 | 47.8 | 27.0 | 0.04 | 16.14 | 1399 |
| torchvision_run1 | 0.8089 | 0.6836 | 0.3425 | 0.5 | 0.484 | 0.344 | 11725 | 20957 | 0.019 | 0.906 | 52.9 | 138.1 | 15.3 | 4.49 | 144.50 | 2134 |
| torchvision_run2 | 0.8722 | 0.8070 | 0.4796 | 0.5 | 0.585 | 0.462 | 10463 | 17203 | 0.092 | 0.934 | 42.5 | 63.0 | 21.4 | 2.18 | 144.50 | 2819 |
| yunet | 0.8875 | 0.8719 | 0.7696 | 0.6 | 0.873 | 0.689 | 3201 | 9936 | 0.484 | 0.908 | 8.7 | 16.2 | 98.9 | 0.01 | 0.22 | 1206 |

## Detector fine-tuning

device mps, 23.4 min, train {'images': 6000, 'boxes_used': 59600, 'boxes_flagged_ignore': 1146, 'boxes_below_min_size': 9857}, val {'images': 300, 'boxes_used': 2985, 'boxes_flagged_ignore': 46, 'boxes_below_min_size': 408}

| epoch | steps | train loss | val_ft AP | minutes |
|---|---|---|---|---|
| 0 | 1500 | 0.5265 | 0.4548 | 12.3 |
| 1 | 3000 | 0.4873 | 0.4897 | 11.0 |

## Recognition - LFW

| embedder | 10-fold acc | AUC | EER | TAR@FAR1e-3 | TAR@FAR1e-2 | thr (FPIR 1% calib) | FAR @thr | FRR @thr | rank-1 | DIR | FPIR | unknown reject | mis-ID | ms/face (b1 p50) | p95 | size MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arcface_r50 | 0.9985 ± 0.0017 | 0.99931 | 0.0018 | 0.9973 | 0.9983 | 0.298 | 0.0000 | 0.0050 | 0.9988 | 0.9982 | 0.0078 | 0.9922 | 0.0000 | 16.91 | 17.41 | 166.3 |
| sface | 0.9935 ± 0.0037 | 0.99877 | 0.0082 | 0.9880 | 0.9927 | 0.464 | 0.0000 | 0.0583 | 0.9958 | 0.9867 | 0.0099 | 0.9901 | 0.0000 | 3.77 | 3.90 | 36.9 |

## Export

```json
{
 "yunet": {
  "onnx": "/Users/ssrivastava/Personal/face_ai/models/exports/onnx/face_detection_yunet_2023mar.onnx",
  "onnx_size_mb": 0.22181415557861328,
  "coreml": {
   "mlpackage": "/Users/ssrivastava/Personal/face_ai/models/exports/coreml/yunet.mlpackage",
   "convert_seconds": 1.8567500114440918,
   "size_mb": 0.18761348724365234,
   "parity_vs_onnxruntime": {
    "samples": 8,
    "max_abs_diff": 0.05648016929626465,
    "note": "Core ML run on macOS with FLOAT16 compute precision"
   }
  }
 },
 "sface": {
  "onnx": "/Users/ssrivastava/Personal/face_ai/models/exports/onnx/face_recognition_sface_2021dec.onnx",
  "onnx_size_mb": 36.90371799468994,
  "coreml": {
   "mlpackage": "/Users/ssrivastava/Personal/face_ai/models/exports/coreml/sface.mlpackage",
   "convert_seconds": 2.229825973510742,
   "size_mb": 18.428502082824707,
   "parity_vs_onnxruntime": {
    "samples": 8,
    "max_abs_diff": 0.00547868013381958,
    "min_cosine": 0.9999975562095642,
    "note": "Core ML run on macOS with FLOAT16 compute precision"
   }
  }
 },
 "mediapipe_face_landmarker": {
  "note": "Not converted: use the official MediaPipe Tasks iOS SDK (MediaPipeTasksVision) with the same face_landmarker.task file; connection topology is identical (FaceLandmarksConnections)."
 }
}
```
