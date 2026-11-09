"""Local browser UI: upload an image and see detections, identities and the face mesh.

    python -m face_ai.api.web_ui [--port 8000] [--open]      # or simply ./run.sh

Binds to 127.0.0.1 by default. Uploaded images are processed in memory and never written to disk;
only explicit enrollments (with a consent confirmation) are saved, to the private gallery.
Standard library HTTP server, no external services or CDNs.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import logging
import re
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from face_ai.api.schemas import MATCH, NO_MATCH, UNCERTAIN_LABEL, IdentityMatch
from face_ai.api.service import FaceAIService
from face_ai.config import GALLERY_DIR, PRETRAINED_DIR, RAW_DIR, load_config, resolve_path
from face_ai.detection.detector import create_detector
from face_ai.detection.tracking import IoUTracker
from face_ai.landmarks import topology
from face_ai.logging_config import setup_logging
from face_ai.recognition.gallery import ConsentRecord, GalleryError, clean_profile
from face_ai.recognition.calibration import format_rate
from face_ai.recognition.temporal import TrackIdentityAggregator
from face_ai.visualization.overlay import OverlayOptions

log = logging.getLogger("face_ai.web_ui")

STATIC_DIR = Path(__file__).with_name("static")
_TASKS_VISION = PRETRAINED_DIR / "tasks_vision" / "package"
VENDOR_FILES = {  # the only model files the page may fetch: the in-page face mesh runtime
    "/vendor/vision_bundle.mjs": (_TASKS_VISION / "vision_bundle.mjs", "text/javascript"),
    "/vendor/wasm/vision_wasm_internal.js": (_TASKS_VISION / "wasm/vision_wasm_internal.js", "text/javascript"),
    "/vendor/wasm/vision_wasm_internal.wasm": (_TASKS_VISION / "wasm/vision_wasm_internal.wasm", "application/wasm"),
    "/vendor/wasm/vision_wasm_nosimd_internal.js": (_TASKS_VISION / "wasm/vision_wasm_nosimd_internal.js", "text/javascript"),
    "/vendor/wasm/vision_wasm_nosimd_internal.wasm": (_TASKS_VISION / "wasm/vision_wasm_nosimd_internal.wasm", "application/wasm"),
    "/vendor/face_landmarker.task": (PRETRAINED_DIR / "face_landmarker" / "face_landmarker.task", "application/octet-stream"),
}
MAX_BODY_BYTES = 60 * 2**20
MAX_RETURN_SIDE = 2400  # annotated image is downscaled for the browser above this size
SFX_PATH_RE = re.compile(r"/sfx/([a-z0-9]+\.wav)")
PERSON_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\- ]{0,63}$")
SAMPLES = {
    "group": "12--Group/12_Group_Group_12_Group_Group_12_10.jpg",
    "crowd": "2--Demonstration/2_Demonstration_Demonstration_Or_Protest_2_58.jpg",
}


VIDEO_SESSION_TTL_S = 120
MAX_VIDEO_SESSIONS = 8


class BadRequest(ValueError):
    pass


class VideoSession:
    """Per-browser-tab video state: temporary track ids and per-track embedding history."""

    def __init__(self, window: int):
        self.tracker = IoUTracker(iou_threshold=0.3, max_misses=15)
        self.aggregator = TrackIdentityAggregator(window=window)
        self.last_seen = time.time()
        self.frames = 0


def decode_data_url(data_url: str) -> np.ndarray:
    try:
        b64 = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
        buf = np.frombuffer(base64.b64decode(b64, validate=False), np.uint8)
    except (ValueError, IndexError) as e:
        raise BadRequest(f"invalid image data: {e}") from e
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise BadRequest("could not decode image (use JPEG, PNG, WebP or BMP)")
    return img


def encode_jpeg(img: np.ndarray, quality: int = 90) -> str:
    h, w = img.shape[:2]
    if max(h, w) > MAX_RETURN_SIDE:
        s = MAX_RETURN_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class FaceUI:
    """Holds the loaded models and gallery; all model calls run under one lock."""

    def __init__(self, gallery_dir: Path):
        self.lock = threading.Lock()
        self.det_cfg = load_config("detection")
        self.svc = FaceAIService.from_configs()
        self.gallery_dir = gallery_dir
        if (gallery_dir / "gallery.json").exists():
            self.svc.load_gallery(gallery_dir)
        self._detectors: dict[str, Any] = {}
        self.default_id_threshold = self.svc.matcher.threshold
        self.video_sessions: dict[str, VideoSession] = {}
        self.video_window = int(self.svc.recognition_cfg.get("video", {}).get("aggregate_window", 15))
        _ = self.svc.embedder, self.svc.landmarker  # load up front so the first request is fast

    def available_detectors(self) -> list[str]:
        out = ["yunet"]
        if resolve_path(self.det_cfg["scrfd"]["model_path"]).exists():
            out.append("scrfd")
        if resolve_path(self.det_cfg["torchvision"]["checkpoint"]).exists():
            out.append("torchvision")
        return out

    def detector(self, name: str):
        if name not in self.available_detectors():
            raise BadRequest(f"detector {name!r} is not available")
        if name not in self._detectors:
            self._detectors[name] = create_detector({**self.det_cfg, "backend": name})
        return self._detectors[name]

    def status(self) -> dict[str, Any]:
        g = self.svc.gallery
        sample_root = RAW_DIR / "wider_face" / "WIDER_val" / "images"
        return {
            "detectors": self.available_detectors(),
            "default_detector": self.det_cfg.get("backend", "yunet"),
            "det_thresholds": {d: self.det_cfg.get(d, {}).get("score_threshold", 0.5) for d in ("yunet", "scrfd", "torchvision")},
            "embedder": self.svc.embedder.name,
            "flip_augment": self.svc.embedder.flip_augment,
            "keypoint_refiner": self.svc.keypoint_refiner is not None,
            "id_threshold": self.default_id_threshold,
            "id_reject_threshold": self.svc.matcher.reject,
            "verify_threshold": self.svc.matcher.verify_threshold,
            "margin": self.svc.matcher.margin,
            "calibrated": self.svc.matcher.calibration is not None,
            "min_quality": self.svc.quality_cfg.min_quality,
            "gallery": [{"person_id": p, "references": g.get(p).num_references, "profile": g.get(p).profile}
                        for p in g.people()],
            "samples": [k for k, v in SAMPLES.items() if (sample_root / v).exists()],
            "mesh_groups": topology.available_groups(),
            "default_mesh_groups": list(topology.DEFAULT_GROUPS),
            # for drawing the mesh client-side over live video
            "live_mesh": all(path.is_file() for path, _ in VENDOR_FILES.values()),
            "mesh_connections": {g: topology.connections(g) for g in topology.available_groups()},
            "mesh_colors": {g: "#%02x%02x%02x" % tuple(reversed(c)) for g, c in topology.DEFAULT_COLORS.items()},
            "mesh_triangles": topology.triangles(),
            "video_window": self.video_window,
        }

    def sample(self, name: str) -> bytes:
        if name not in SAMPLES:
            raise BadRequest("unknown sample")
        return (RAW_DIR / "wider_face" / "WIDER_val" / "images" / SAMPLES[name]).read_bytes()

    def analyze(self, req: dict[str, Any]) -> dict[str, Any]:
        img = decode_data_url(req.get("image", ""))
        det = self.detector(req.get("detector", "yunet"))
        det.score_threshold = float(req.get("det_threshold", det.score_threshold))
        id_threshold = float(req.get("id_threshold", self.default_id_threshold))
        mesh = bool(req.get("mesh", True))
        groups = [g for g in req.get("mesh_groups", topology.DEFAULT_GROUPS) if g in topology.available_groups()]
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        dets = det.detect(img)
        timings["detect_ms"] = (time.perf_counter() - t0) * 1000

        gallery = self.svc.gallery
        identities, qualities, verifications = None, None, None
        verify_pid = req.get("verify_person") or None
        if verify_pid is not None and verify_pid not in gallery:
            raise BadRequest("the person to verify against is not enrolled")
        if dets and len(gallery):
            t0 = time.perf_counter()
            crops = [self.svc.align(img, d) for d in dets]
            vecs = self.svc.embedder.embed(crops)
            qualities = [self.svc.assess_quality(d, c) for d, c in zip(dets, crops)]
            matcher = dataclasses.replace(self.svc.matcher, threshold=id_threshold)
            identities = matcher.identify_batch(vecs, gallery, qualities)
            if verify_pid:
                verifications = [matcher.verify_against(v, verify_pid, gallery, q) for v, q in zip(vecs, qualities)]
            timings["identify_ms"] = (time.perf_counter() - t0) * 1000

        landmarks = None
        if mesh and dets:
            t0 = time.perf_counter()
            landmarks = [self.svc.extract_landmarks(img, d) for d in dets]
            timings["mesh_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        opts = OverlayOptions(show_mesh=mesh, mesh_groups=groups, show_keypoints=bool(req.get("keypoints")),
                              show_index=True, show_detection_score=bool(req.get("show_scores", True)))
        shown = identities
        if verifications:  # verify mode: label faces with the 1:1 answer, not the 1:N one
            shown = [IdentityMatch(v.person_id if v.is_match else None, v.similarity, v.threshold, decision=v.decision,
                                   display_label={MATCH: v.person_id, NO_MATCH: f"not {v.person_id}"}.get(v.decision, UNCERTAIN_LABEL))
                     for v in verifications]
        # client_render: the page draws its own HUD over the image it already has, so skip the server render
        client_render = bool(req.get("client_render"))
        vis = None if client_render else self.svc.render_overlay(img, dets, landmarks, shown, opts)
        timings["render_ms"] = (time.perf_counter() - t0) * 1000

        cal = self.svc.matcher.calibration
        bound = 1.0 / cal.n_impostor if cal else None
        faces = []
        for i, d in enumerate(dets):
            ident = identities[i] if identities else None
            q = qualities[i] if qualities else None
            ver = verifications[i] if verifications else None
            faces.append({
                "index": i + 1,
                "det_score": round(d.score, 4),
                "box": [round(v, 1) for v in (d.bbox.x1, d.bbox.y1, d.bbox.width, d.bbox.height)],
                "size_px": round(min(d.bbox.width, d.bbox.height)),
                "identity": ident.label if ident else None,
                "decision": ident.decision if ident else None,
                "reason": ident.reason if ident else None,
                "similarity": None if ident is None or ident.similarity is None else round(ident.similarity, 4),
                "margin": None if ident is None or ident.margin is None else round(ident.margin, 4),
                # the stranger-chance only means something for a claimed match; hide it for "unknown"
                "false_match": None if ident is None or ident.estimated_far is None or ident.decision == "unknown"
                else format_rate(ident.estimated_far, bound),
                "candidates": [[c, round(s, 4)] for c, s in ident.candidates] if ident else [],
                "quality": None if q is None else {"score": round(q.score, 3), "acceptable": q.acceptable,
                                                   "limiting": q.limiting, "issues": q.issues},
                "verify": None if ver is None else {
                    "person_id": ver.person_id, "decision": ver.decision, "similarity": round(ver.similarity, 4),
                    "threshold": ver.threshold, "reject_threshold": ver.reject_threshold, "reason": ver.reason,
                    "false_match": format_rate(ver.estimated_far, bound)
                    if ver.estimated_far is not None and ver.decision != "unknown" else None},
                "mesh": None if landmarks is None else landmarks[i] is not None,
                "profile": self._profile_for(ver.person_id if ver is not None and ver.is_match
                                             else ident.person_id if ident is not None and ident.is_known else None),
                "mesh_points": None if not client_render or landmarks is None or landmarks[i] is None
                else np.round(landmarks[i].points[:, :3], 1).ravel().tolist(),  # x, y, z per point
            })
        return {
            "image": None if vis is None else encode_jpeg(vis),
            "width": img.shape[1], "height": img.shape[0],
            "faces": faces, "timings": timings,
            "detector": det.name, "det_threshold": det.score_threshold,
            "embedder": self.svc.embedder.name, "id_threshold": id_threshold,
            "id_reject_threshold": min(id_threshold, self.svc.matcher.reject),
            "gallery_size": len(gallery), "verify_person": verify_pid,
        }

    def _video_session(self, sid: str) -> VideoSession:
        now = time.time()
        for k in [k for k, v in self.video_sessions.items() if now - v.last_seen > VIDEO_SESSION_TTL_S]:
            del self.video_sessions[k]
        if sid not in self.video_sessions:
            if len(self.video_sessions) >= MAX_VIDEO_SESSIONS:
                oldest = min(self.video_sessions, key=lambda k: self.video_sessions[k].last_seen)
                del self.video_sessions[oldest]
            self.video_sessions[sid] = VideoSession(self.video_window)
        sess = self.video_sessions[sid]
        sess.last_seen = now
        return sess

    def video_frame(self, req: dict[str, Any]) -> dict[str, Any]:
        """One frame of a live feed. Returns JSON only (the page draws the overlay on the video)."""
        sid = str(req.get("session", ""))[:64]
        if not sid:
            raise BadRequest("missing session id")
        sess = self._video_session(sid)
        img = decode_data_url(req.get("image", ""))
        det = self.detector(req.get("detector", "yunet"))
        det.score_threshold = float(req.get("det_threshold", det.score_threshold))
        id_threshold = float(req.get("id_threshold", self.default_id_threshold))
        aggregate = bool(req.get("aggregate", True))
        verify_pid = req.get("verify_person") or None
        gallery = self.svc.gallery
        if verify_pid is not None and verify_pid not in gallery:
            raise BadRequest("the person to verify against is not enrolled")
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        dets = det.detect(img)
        sess.tracker.update(dets)
        timings["detect_ms"] = (time.perf_counter() - t0) * 1000

        idents: list[Any] = [None] * len(dets)
        verifs: list[Any] = [None] * len(dets)
        used = [0] * len(dets)
        quals: list[Any] = [None] * len(dets)
        if dets and len(gallery):
            t0 = time.perf_counter()
            crops = [self.svc.align(img, d) for d in dets]
            vecs = self.svc.embedder.embed(crops)
            matcher = dataclasses.replace(self.svc.matcher, threshold=id_threshold)
            for i, (d, c, v) in enumerate(zip(dets, crops, vecs)):
                q = self.svc.assess_quality(d, c)
                if aggregate and q is not None and d.track_id is not None:
                    v, q, used[i] = sess.aggregator.update(d.track_id, v, q)
                quals[i] = q
                idents[i] = matcher.identify(v, gallery, q)
                if verify_pid:
                    verifs[i] = matcher.verify_against(v, verify_pid, gallery, q)
            timings["identify_ms"] = (time.perf_counter() - t0) * 1000
        sess.aggregator.prune(sess.tracker.active_ids())

        meshes: list[Any] = [None] * len(dets)
        if req.get("mesh", True) and dets:
            t0 = time.perf_counter()
            for i, d in enumerate(dets):
                lm = self.svc.extract_landmarks(img, d)
                meshes[i] = None if lm is None else np.round(lm.points[:, :3], 1).ravel().tolist()  # x, y, z
            timings["mesh_ms"] = (time.perf_counter() - t0) * 1000
        sess.frames += 1

        faces = []
        for i, d in enumerate(dets):
            ident, ver, q = idents[i], verifs[i], quals[i]
            dec = ver if ver is not None else ident
            if ver is not None:
                label = {"match": ver.person_id, "unknown": f"not {ver.person_id}"}.get(ver.decision, "Uncertain")
            else:
                label = ident.label if ident else None
            faces.append({
                "track_id": d.track_id,
                "box": [round(v, 1) for v in (d.bbox.x1, d.bbox.y1, d.bbox.width, d.bbox.height)],
                "det_score": round(d.score, 3),
                "label": label,
                "decision": dec.decision if dec else None,
                "similarity": None if dec is None or dec.similarity is None else round(dec.similarity, 3),
                "reason": dec.reason if dec else None,
                "frames_used": used[i],
                "profile": self._profile_for(ver.person_id if ver is not None and ver.is_match
                                             else ident.person_id if ident is not None and ident.is_known else None),
                "quality": None if q is None else {"score": round(q.score, 2), "issues": q.issues},
                "mesh": meshes[i],
            })
        return {"faces": faces, "width": img.shape[1], "height": img.shape[0], "timings": timings,
                "frame": sess.frames, "gallery_size": len(gallery), "verify_person": verify_pid,
                "id_threshold": id_threshold}

    def video_reset(self, req: dict[str, Any]) -> dict[str, Any]:
        self.video_sessions.pop(str(req.get("session", ""))[:64], None)
        return {"ok": True}

    def _profile_for(self, person_id: str | None) -> dict[str, str] | None:
        if person_id is None or person_id not in self.svc.gallery:
            return None
        return {"person_id": person_id, **self.svc.gallery.get(person_id).profile}

    @staticmethod
    def _slug(text: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_.\- ]+", "", text).strip().lower().replace(" ", "_")
        return s[:64]

    def enroll_face(self, req: dict[str, Any]) -> dict[str, Any]:
        """Register (or add a reference for) the face at ``box`` in ``image``, with an optional profile card."""
        if req.get("consent_confirmed") is not True:
            raise BadRequest("confirm that this person agreed to be enrolled")
        profile = clean_profile(req.get("profile"))
        pid = str(req.get("person_id") or "").strip() or self._slug(profile.get("display_name", ""))
        if not PERSON_ID_RE.match(pid):
            raise BadRequest("enter a name or person id (letters, digits, space, '_', '-', '.')")
        add_ref = bool(req.get("add_reference"))
        exists = pid in self.svc.gallery
        if exists and not add_ref:
            raise BadRequest(f"{pid!r} is already enrolled; use a different id or add this photo as a reference")
        if add_ref and not exists:
            raise BadRequest(f"{pid!r} is not enrolled")
        by = str(req.get("consent_granted_by") or profile.get("display_name") or pid).strip()
        purpose = str(req.get("purpose") or "personal recognition").strip()
        img = decode_data_url(req.get("image", ""))
        try:
            x, y, w, h = (float(v) for v in req["box"])
        except (KeyError, TypeError, ValueError) as e:
            raise BadRequest("missing face box") from e
        side = max(w, h) * 1.6  # square crop around the clicked face, padded at the image border
        cx, cy = x + w / 2, y + h / 2
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        s_ = int(round(side))
        H, W = img.shape[:2]
        pad = max(0, -x0, -y0, x0 + s_ - W, y0 + s_ - H)
        src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else img
        crop = src[y0 + pad:y0 + pad + s_, x0 + pad:x0 + pad + s_]
        if crop.size == 0:
            raise BadRequest("face box is outside the image")
        consent = ConsentRecord(True, by, purpose)
        report = self.svc.enroll_person(pid, [crop], consent, min_face_size=40)
        if report.ok:
            if profile or not exists:
                self.svc.gallery.set_profile(pid, {**(self.svc.gallery.get(pid).profile if exists else {}), **profile})
            self.svc.save_gallery(self.gallery_dir)
        return {"person_id": pid, "accepted": report.accepted,
                "rejected": [{"image": "selected face", "reason": why} for _, why in report.rejected],
                "warnings": report.warnings, "status": self.status()}

    def update_profile(self, req: dict[str, Any]) -> dict[str, Any]:
        pid = str(req.get("person_id", ""))
        if pid not in self.svc.gallery:
            raise BadRequest("not enrolled")
        profile = self.svc.gallery.set_profile(pid, req.get("profile"))
        self.svc.save_gallery(self.gallery_dir)
        return {"person_id": pid, "profile": profile, "status": self.status()}

    def enroll(self, req: dict[str, Any]) -> dict[str, Any]:
        pid = str(req.get("person_id", "")).strip()
        if not PERSON_ID_RE.match(pid):
            raise BadRequest("person id: 1-64 letters, digits, space, '_', '-' or '.'")
        if req.get("consent_confirmed") is not True:
            raise BadRequest("confirm that this person agreed to be enrolled")
        by, purpose = str(req.get("consent_granted_by", "")).strip(), str(req.get("purpose", "")).strip()
        if not by or not purpose:
            raise BadRequest("consent requires who agreed and for what purpose")
        images = [decode_data_url(x) for x in req.get("images", [])][:20]
        if not images:
            raise BadRequest("add at least one photo")
        retention = req.get("retention_days")
        consent = ConsentRecord(True, by, purpose, retention_days=int(retention) if retention else None)
        report = self.svc.enroll_person(pid, images, consent)
        if report.ok:
            self.svc.save_gallery(self.gallery_dir)
        return {"person_id": pid, "accepted": report.accepted,
                "rejected": [{"image": ref, "reason": why} for ref, why in report.rejected],
                "warnings": report.warnings, "status": self.status()}

    def remove(self, req: dict[str, Any]) -> dict[str, Any]:
        pid = str(req.get("person_id", ""))
        if not self.svc.gallery.remove(pid):
            raise BadRequest("not enrolled")
        self.svc.save_gallery(self.gallery_dir)
        return {"removed": pid, "status": self.status()}


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def make_handler(ui: FaceUI, allowed_hosts: frozenset[str] | None = LOOPBACK_HOSTS):
    """``allowed_hosts``: hostnames accepted in the Host header (blocks DNS rebinding); None accepts any."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "face_ai"

        def log_message(self, fmt: str, *args: Any) -> None:  # no request logging (may contain ids)
            pass

        def _host_ok(self) -> bool:
            host = self.headers.get("Host", "")
            return allowed_hosts is None or urlparse(f"//{host}").hostname in allowed_hosts

        def _same_origin(self) -> bool:
            """Cross-site pages may not drive the API: the Origin (when sent) must be this server, and the
            body must be declared JSON, which a cross-site page can only send after a CORS preflight we never answer."""
            origin = self.headers.get("Origin")
            if origin is not None and urlparse(origin).netloc != self.headers.get("Host", ""):
                return False
            return self.headers.get("Content-Type", "").split(";")[0].strip().lower() == "application/json"

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            if not self._host_ok():
                return self._json(403, {"error": "unexpected Host header"})
            url = urlparse(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/status":
                    with ui.lock:
                        self._json(200, ui.status())
                elif url.path == "/api/sample":
                    self._send(200, ui.sample(parse_qs(url.query).get("name", [""])[0]), "image/jpeg")
                elif (sound := SFX_PATH_RE.fullmatch(url.path)) and (STATIC_DIR / "sfx" / sound[1]).is_file():
                    self._send(200, (STATIC_DIR / "sfx" / sound[1]).read_bytes(), "audio/wav")
                elif url.path in VENDOR_FILES and VENDOR_FILES[url.path][0].is_file():
                    path, ctype = VENDOR_FILES[url.path]
                    self._send(200, path.read_bytes(), ctype)
                else:
                    self._json(404, {"error": "not found"})
            except BadRequest as e:
                self._json(400, {"error": str(e)})

        def do_POST(self) -> None:
            routes = {"/api/analyze": ui.analyze, "/api/enroll": ui.enroll, "/api/remove": ui.remove,
                      "/api/video/frame": ui.video_frame, "/api/video/reset": ui.video_reset,
                      "/api/enroll_face": ui.enroll_face, "/api/profile": ui.update_profile}
            if not self._host_ok() or not self._same_origin():
                return self._json(403, {"error": "cross-site request refused"})
            fn = routes.get(urlparse(self.path).path)
            if fn is None:
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY_BYTES:
                return self._json(413 if n > 0 else 400, {"error": f"request body must be 1 byte - {MAX_BODY_BYTES // 2**20} MB"})
            try:
                req = json.loads(self.rfile.read(n))
                with ui.lock:
                    self._json(200, fn(req))
            except (BadRequest, GalleryError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - report to the page, keep serving
                log.exception("request failed")
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 unless you understand the privacy impact")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gallery", default=str(GALLERY_DIR))
    ap.add_argument("--open", action="store_true", help="open the UI in the default browser")
    ap.add_argument("--no-open", action="store_true", help="overrides --open")
    args = ap.parse_args()
    setup_logging()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("serving on %s: anyone who can reach this address can upload images and edit the gallery", args.host)
    log.info("loading models ...")
    ui = FaceUI(Path(args.gallery))
    # bound to a wildcard address the server is reached by LAN names we can't know, so skip the Host check there
    allowed = None if args.host in ("0.0.0.0", "::", "") else LOOPBACK_HOSTS | {args.host}
    server = ThreadingHTTPServer((args.host, args.port), make_handler(ui, allowed))
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '::1') else args.host}:{args.port}/"
    log.info("face_ai UI ready at %s  (Ctrl+C to stop)", url)
    if args.open and not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        ui.svc.close()


if __name__ == "__main__":
    main()
