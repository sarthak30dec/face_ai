# Datasets: selection, provenance, validation

Selection date: 2026-10-30. Everything in this document was checked against the live hosts on that date; the
manifests in `data/manifests/` hold the machine-readable record (URLs, sizes, SHA-256, counts, validation).

## 1. What each task actually needs

The three components need different kinds of data, so they are not compared as interchangeable:

| Task | Needs | Role in this project |
|---|---|---|
| Face detection | Many images with *all* faces boxed, wide range of scale/pose/occlusion, an official evaluation protocol | Baseline evaluation + bounded fine-tuning |
| Face recognition | Identity labels. *Training* needs 10⁴–10⁶ identities; *evaluation* needs a standard pairs/identification protocol | Evaluation + threshold calibration only (see §3) |
| Landmark mesh | Dense 3D landmark labels | None: MediaPipe's pretrained model is used as-is (visualization only) |

## 2. Face detection

| Candidate | Size | Annotation | Protocol | License / access (checked) | Verdict |
|---|---|---|---|---|---|
| **WIDER FACE** | 32,203 images, 393,703 faces, 61 event classes | Boxes + blur/expression/illumination/occlusion/pose/invalid flags | Official easy/medium/hard AP (PASCAL VOC-style), eval code + GT `.mat` published | CC BY-NC-ND; official page links Google Drive, Tencent Weiyun **and Hugging Face** | **Selected** |
| FDDB | 2,845 images, 5,171 faces | Ellipses | Continuous/discrete ROC | Hosted on `vis-www.cs.umass.edu`, which returned NXDOMAIN | Too small, ellipse GT, unreachable |
| AFW / PASCAL Face | 205 / 851 images | Boxes | AP | Small legacy benchmarks | Too small to be decisive |
| MALF | 5,250 images | Boxes + attributes | Fine-grained | Older, limited availability | Smaller, less used today |
| IJB-A/B/C | — | Boxes + identities | NIST protocols | Requires a NIST access request | Needs an action only the user can take |

**Why WIDER FACE:** it is the de facto benchmark that every modern face detector (YuNet, SCRFD, RetinaFace) reports on,
so our baseline numbers are directly comparable with published ones. It has a reproducible official evaluation
(we ported `eval_tools` and reproduce published APs to within ~0.01; see `docs/evaluation.md`), dense annotation of tiny
and occluded faces, and an authorized download that needs no login.

**Download:** the official homepage (`http://shuoyang1213.me/WIDERFACE/`) links the Hugging Face copy
(`huggingface.co/datasets/CUHK-CSE/wider_face`) next to Google Drive/Weiyun, so we use it: it supports HTTP range requests
(resumable) and publishes SHA-256 (`x-linked-etag`), which the downloader verifies. Annotations (`wider_face_split.zip`)
and evaluation tools (`eval_tools.zip`) come from the official homepage itself; their hashes were recorded on first
download and pinned in `configs/datasets.yaml`.

**Not downloaded:** `WIDER_test.zip` (1.84 GB). Test labels are not public (evaluation is by submission to the
authors), so it adds nothing locally. `--include-optional` fetches it.

**Split policy** (`data/splits/wider_face/`, seed 42):

| Split | Source | Images | Used for |
|---|---|---|---|
| `train_ft` | official train, 90 % per event class | 11,590 | fine-tuning |
| `val_ft` | official train, 10 % per event class | 1,289 | checkpoint selection |
| `test` | official **val** | 3,226 | all reported detection numbers; never used for training or selection |

One train image is byte-identical to a val image
(`29--Students_Schoolkids/29_Students_Schoolkids_Students_Schoolkids_29_475.jpg`); it is excluded from
`train_ft`/`val_ft` to prevent leakage and listed in `split_info.json`.

## 3. Face recognition

### Should we train a recognition network at all?

No. State-of-the-art embedders are trained on 0.5–4 M identities with margin losses (ArcFace, CosFace, SFace).
Reaching their accuracy needs both that data and days of multi-GPU training. A model trained locally on what is
legally available would be far *worse* than the pretrained ones. Enrolling a new person only adds embeddings to the
gallery, with no retraining (the project requirement). So recognition **training** is not justified. The work goes into
evaluation and threshold calibration instead.

### Candidates

| Candidate | Identities / images | Suitable as | Access (checked or documented status) | Verdict |
|---|---|---|---|---|
| **LFW** | 5,749 / 13,233 | Benchmark (View 2: 6,000 pairs, 10 folds); too few multi-image identities to train | Official host down (NXDOMAIN); scikit-learn's figshare mirror with SHA-256 pinned in scikit-learn source | **Selected (evaluation + calibration)** |
| CelebA | 10,177 / 202,599 | Possible small-scale training/eval | Non-commercial; distributed through Google Drive folders with quota limits | Not needed (no training); drive quotas make automation unreliable |
| VGGFace2 | 9,131 / 3.31 M | Training | Withdrawn by the authors; no authorized download | Rejected: no authorized source |
| MS-Celeb-1M (MS1M) | ~100 k / ~10 M | Training | Withdrawn by Microsoft (2019) over consent concerns | Rejected: withdrawn, consent problems |
| CASIA-WebFace | 10,575 / 494,414 | Training | No official distribution; copies circulate unofficially | Rejected: no authorized source |
| WebFace260M / 42M | 2–4 M ids | Training | Requires a signed agreement with the authors | Needs a user action; also unnecessary |
| CFP-FP, AgeDB, IJB-B/C | — | Harder benchmarks | Access requests / agreements | Needs a user action (next step, §6) |

**Why LFW:** it is the most widely reported verification benchmark, so our numbers can be compared with each model's
published LFW accuracy (a reproduction check on the whole align → embed → compare pipeline). It has an official,
reproducible pairs protocol and enough multi-image identities (1,680 with ≥ 2 images, 901 with ≥ 3) to build an
identity-disjoint open-set identification protocol. It is also small (173 MB).

**LFW's limits, stated plainly:** modern models are near the LFW ceiling (> 99 %), so LFW cannot separate
strong models well at low FAR. It is dominated by frontal, well-lit photos of public figures (mostly male and
lighter-skinned), and its 250×250 crops come from a Viola-Jones detector, which biases it toward detectable faces.
LFW numbers are **not** a guarantee of accuracy on your users, cameras or lighting.

**Download:** `vis-www.cs.umass.edu` returned NXDOMAIN on 2026-10-30. The downloader uses
`ndownloader.figshare.com/files/{5976018,5976006,5976012,5976009}`, the mirror scikit-learn's `fetch_lfw_*`
uses, and verifies the SHA-256 values pinned in `sklearn/datasets/_lfw.py`. All four files matched.

### Recognition protocols (`data/splits/lfw/`, seed 42)

* `pairs_view2.json`: official View 2, 10 folds × (300 same + 300 different). Accuracy uses 10-fold cross-validation,
  with each fold's threshold chosen on the other nine folds.
* `openset.json`: an **identity-disjoint** open-set protocol:
  * All 5,749 identities are shuffled into `calibration` (2,874) and `test` (2,875) halves; no identity is in both.
  * Within each half, identities with ≥ 3 images are *known*: 2 random images form the (in-memory) gallery and up to 10
    **other** images are known probes. Every other identity contributes 1 *unknown* probe.
  * Calibration: 460 enrolled, 1,674 known probes, 2,414 unknown probes. Test: 441 enrolled, 1,656 known, 2,434 unknown.
  * Thresholds are chosen on calibration only; the reported numbers are on test.
  * These galleries are created with `Gallery(..., ephemeral=True)`, which **cannot be saved**. No public-dataset
    identity is ever written into an enrollment gallery.

## 4. What was downloaded (verified)

| Dataset | Files (verified) | Archive size | Extracted | Location | Counts |
|---|---|---|---|---|---|
| WIDER FACE | `WIDER_train.zip` (SHA-256 ✓), `WIDER_val.zip` (SHA-256 ✓), `wider_face_split.zip`, `eval_tools.zip` (hash recorded + pinned) | 1.7 GB | ~1.8 GB | `data/raw/wider_face/` | train 12,880 imgs / 159,420 faces; val 3,226 imgs / 39,708 faces |
| LFW | `lfw.tgz`, `pairs.txt`, `pairsDevTrain.txt`, `pairsDevTest.txt` (all SHA-256 ✓) | 172 MB | ~200 MB | `data/raw/lfw/` | 13,233 images, 5,749 identities, 6,000 pairs |

Archives are kept in `data/raw/<dataset>/archives/` so re-runs can verify integrity and skip downloading. Extraction
rejects absolute paths, `..` components, symlinks and device files, and the zip CRCs are tested before extraction.

## 5. Validation results (`scripts/validate_datasets.py`)

Full report: `outputs/reports/dataset_validation.md`. JSON: `data/manifests/*_validation.json`.

| Check | WIDER train | WIDER val | LFW |
|---|---|---|---|
| Annotated images present on disk | 12,880 / 12,880 | 3,226 / 3,226 | 13,233 |
| Corrupt / undecodable images | 0 | 0 | 0 |
| Byte-identical duplicate groups (within split) | 2 (kept, reported) | 0 | 2 (kept, reported) |
| Degenerate boxes (w or h ≤ 0) | 27 | 11 | — |
| Boxes flagged `invalid` by WIDER | 2,399 | 585 | — |
| Boxes extending past the image border | 1 | 1 | — |
| Boxes < 4 px | 1,456 | 440 | — |
| Eval `.mat` vs txt face count | — | 39,708 = 39,708 ✓ | — |
| Train ∩ val identical files | 1 (excluded from fine-tuning) | | — |
| pairs.txt references present | — | — | 6,000 pairs, 0 missing |

Nothing is silently dropped. The COCO conversion (`data/processed/wider_face/*_coco.json`) keeps every source box and
its attributes. Degenerate/invalid boxes are marked `iscrowd=1` with an `ignore_reason`, and the trainer reports how
many it skipped. `__MACOSX/` (macOS metadata inside `eval_tools.zip`) is noted and ignored.

LFW face pre-detection (`data/processed/lfw/faces_<detector>.json`): YuNet found the central face in 13,233/13,233
images. SCRFD found 13,226; for the remaining 7 it falls back to the mean LFW keypoints, flagged
`fallback_mean_keypoints`. The reported recognition metrics align with YuNet keypoints, so they include no fallbacks.

## 6. Licensing and use

* **WIDER FACE:** CC BY-NC-ND (non-commercial). Use for research/evaluation. Shipping weights fine-tuned on it in a
  commercial product needs legal review.
* **LFW:** no formal license; a public research benchmark of public figures. Used only for evaluation. Images are never
  enrolled, redistributed or committed.
* Raw datasets are git-ignored. Only manifests, splits and metrics (no images) are tracked.

## 7. Possible additions (each needs an action only you can take)

* CFP-FP and AgeDB (pose and age robustness): request access from the authors.
* IJB-B/IJB-C (template-based 1:N with low-FAR operating points): NIST access request.
* Most importantly, a **consented in-domain set**: images from your actual cameras/users, collected with consent,
  for threshold calibration before deployment.
