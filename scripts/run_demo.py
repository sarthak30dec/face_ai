#!/usr/bin/env python
"""Face detection + identification + wireframe mesh demo for images, folders and videos.

    python scripts/run_demo.py --input photo.jpg --output outputs/visualizations/photo_annotated.jpg
    python scripts/run_demo.py --input clip.mp4 --output out.mp4 --gallery gallery/ --id-threshold 0.5
    python scripts/run_demo.py --input photos/ --output outputs/visualizations/batch/ --no-mesh --json

Box colour: green = enrolled identity accepted, orange = Unknown, grey = Uncertain (low quality or
ambiguous), blue = no gallery loaded. In video, identity is decided on each track's recent frames.
Label lines: identity (or Unknown) / "sim" = cosine similarity to the best enrolled person /
"det" = detector confidence that this is a face / "track #n" = temporary per-video id.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from face_ai.api.service import FaceAIService
from face_ai.config import GALLERY_DIR, VIS_DIR
from face_ai.detection.tracking import IoUTracker
from face_ai.recognition.temporal import TrackIdentityAggregator
from face_ai.landmarks import topology
from face_ai.logging_config import setup_logging
from face_ai.utils.image_io import is_image, is_video, list_images, load_image, save_image
from face_ai.visualization.overlay import OverlayOptions


def annotate(svc: FaceAIService, frame, opts: OverlayOptions, mesh: bool, identify: bool, tracker=None,
             aggregator: TrackIdentityAggregator | None = None):
    results = svc.analyze(frame, with_landmarks=mesh, with_identity=identify, keep_embeddings=aggregator is not None)
    dets = [r.detection for r in results]
    if tracker is not None:
        tracker.update(dets)
    if aggregator is not None and identify:
        # Decide on the quality-weighted mean embedding of each track's recent frames.
        for r in results:
            if r.embedding is not None and r.quality is not None and r.detection.track_id is not None:
                agg, q, _ = aggregator.update(r.detection.track_id, r.embedding.vector, r.quality)
                r.identity = svc.matcher.identify(agg, svc.gallery, q)
        aggregator.prune(tracker.active_ids() if tracker is not None else set())
    vis = svc.render_overlay(frame, dets, [r.landmarks for r in results],
                             [r.identity for r in results] if identify else None, opts)
    return vis, results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image, directory of images, or video file")
    ap.add_argument("--output", help="output image/video path (or directory for folder input)")
    ap.add_argument("--gallery", help=f"enrollment gallery directory (e.g. {GALLERY_DIR}); omit for detection only")
    ap.add_argument("--det-threshold", type=float, help="detector confidence threshold")
    ap.add_argument("--id-threshold", type=float, help="cosine-similarity threshold for accepting an identity")
    ap.add_argument("--detector", choices=["yunet", "scrfd", "torchvision"])
    ap.add_argument("--embedder", choices=["sface", "arcface_r50"])
    ap.add_argument("--no-mesh", action="store_true", help="disable the face-mesh overlay")
    ap.add_argument("--mesh-groups", nargs="*", default=list(topology.DEFAULT_GROUPS),
                    help=f"mesh regions to draw; options: {topology.available_groups()}")
    ap.add_argument("--keypoints", action="store_true", help="also draw the 5 detector keypoints")
    ap.add_argument("--json", action="store_true", help="write per-face results next to the output (no embeddings)")
    ap.add_argument("--max-frames", type=int, help="video: stop after N frames")
    ap.add_argument("--display", action="store_true", help="show a window while processing video (q to quit)")
    ap.add_argument("--per-frame-identity", action="store_true",
                    help="video: decide identity from each frame alone (default: average each track's recent frames)")
    args = ap.parse_args()
    log = setup_logging()

    svc = FaceAIService.from_configs(detector_backend=args.detector, embedder_backend=args.embedder,
                                     detection_threshold=args.det_threshold, identity_threshold=args.id_threshold)
    identify = args.gallery is not None
    if identify:
        g = svc.load_gallery(args.gallery)
        log.info("gallery: %d enrolled people (%s), identity threshold %.3f", len(g), g.model, svc.matcher.threshold)
    opts = OverlayOptions(show_mesh=not args.no_mesh, mesh_groups=args.mesh_groups, show_keypoints=args.keypoints)
    src = Path(args.input)

    if src.is_dir() or is_image(src):
        files = list_images(src) if src.is_dir() else [src]
        out_dir = Path(args.output) if args.output and (src.is_dir() or not Path(args.output).suffix) else None
        for f in files:
            img = load_image(f)
            t0 = time.perf_counter()
            vis, results = annotate(svc, img, opts, not args.no_mesh, identify)
            dt = (time.perf_counter() - t0) * 1000
            if out_dir is not None:
                out = out_dir / f"{f.stem}_annotated.jpg"
            elif args.output:
                out = Path(args.output)
            else:
                out = VIS_DIR / f"{f.stem}_annotated.jpg"
            save_image(out, vis)
            if args.json:
                out.with_suffix(".json").write_text(json.dumps([r.to_dict() for r in results], indent=1))
            log.info("%s: %d faces, %.0f ms -> %s", f.name, len(results), dt, out)
        return 0

    if is_video(src):
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            log.error("cannot open video %s", src)
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = Path(args.output) if args.output else VIS_DIR / f"{src.stem}_annotated.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        tracker = IoUTracker()
        aggregator = None if args.per_frame_identity else TrackIdentityAggregator(
            window=int(svc.recognition_cfg.get("video", {}).get("aggregate_window", 15)))
        per_frame, n, t_total = [], 0, 0.0
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and n >= args.max_frames):
                break
            t0 = time.perf_counter()
            vis, results = annotate(svc, frame, opts, not args.no_mesh, identify, tracker, aggregator)
            t_total += time.perf_counter() - t0
            writer.write(vis)
            if args.json:
                per_frame.append({"frame": n, "faces": [r.to_dict() for r in results]})
            if args.display:
                cv2.imshow("face_ai", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            n += 1
        cap.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()
        if args.json:
            out.with_suffix(".json").write_text(json.dumps(per_frame))
        log.info("%d frames, %.1f ms/frame (%.1f FPS) -> %s", n, 1000 * t_total / max(n, 1), n / max(t_total, 1e-9), out)
        return 0

    log.error("unsupported input %s", src)
    return 1


if __name__ == "__main__":
    sys.exit(main())
