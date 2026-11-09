#!/usr/bin/env python
"""Evaluate face embedders on LFW: verification (View 2, 10-fold) and open-set identification.

    python scripts/evaluate_recognition.py                          # embedders from configs/evaluation.yaml
    python scripts/evaluate_recognition.py --embedders sface --update-config

Threshold calibration uses only the *calibration* identities of data/splits/lfw/openset.json; the
reported identification numbers come from the disjoint *test* identities.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from face_ai.config import CONFIG_DIR, METRICS_DIR, PROCESSED_DIR, RAW_DIR, VIS_DIR, ensure_dirs, load_config, resolve_path
from face_ai.evaluation.benchmarks import PeakRSS, benchmark, model_size_mb
from face_ai.evaluation.identification_metrics import dir_fpir_curve, open_set_metrics, threshold_for_fpir, top1
from face_ai.evaluation.verification_metrics import eer, far_frr, kfold_accuracy, roc, tar_at_far
from face_ai.logging_config import setup_logging
from face_ai.api.schemas import MATCH, NO_MATCH, UNCERTAIN
from face_ai.recognition.alignment import align_face
from face_ai.recognition.calibration import ScoreCalibration
from face_ai.recognition.quality import FaceQuality, QualityConfig, assess_quality
from face_ai.recognition.embedder import create_embedder
from face_ai.recognition.gallery import Gallery
from face_ai.recognition.matcher import Matcher, score_against_gallery
from face_ai.api.schemas import BoundingBox, FaceDetection
from face_ai.utils.device import hardware_info
from face_ai.utils.image_io import load_image


def embed_images(embedder, faces: dict, rel_paths: list[str], cache: Path, batch: int = 64) -> dict[str, np.ndarray]:
    """Align with cached keypoints and embed; cached per (embedder, alignment detector)."""
    if cache.exists():
        with np.load(cache) as z:
            names, vecs = list(z["names"]), z["vectors"]
        have = dict(zip(names, vecs))
        if all(r in have for r in rel_paths):
            return have
    root = RAW_DIR / "lfw" / "lfw"
    out: dict[str, np.ndarray] = {}
    for i in tqdm(range(0, len(rel_paths), batch), desc=f"embed {embedder.name}", mininterval=2.0):
        chunk = rel_paths[i:i + batch]
        crops = [align_face(load_image(root / r), np.array(faces[r]["keypoints"], np.float32)) for r in chunk]
        for r, v in zip(chunk, embedder.embed(crops)):
            out[r] = v
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, names=np.array(list(out)), vectors=np.stack(list(out.values())))
    return out


def build_gallery(name: str, dim: int, gallery_spec: dict[str, list[str]], emb: dict[str, np.ndarray]) -> Gallery:
    g = Gallery(name, dim, ephemeral=True)  # evaluation fixture: cannot be saved
    for pid, imgs in gallery_spec.items():
        g.add(pid, np.stack([emb[x] for x in imgs]), consent=None)
    return g


def plots(name: str, r: dict, ver_roc: dict, curves: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].semilogx(np.clip(ver_roc["fpr"], 1e-4, 1), ver_roc["tpr"], label=f"AUC={ver_roc['auc']:.4f}")
    ax[0].set(title=f"LFW verification ROC - {name}", xlabel="FAR (log)", ylabel="TAR", xlim=(1e-4, 1), ylim=(0.5, 1.005))
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(loc="lower right")
    ax[1].semilogx(np.clip(curves["fpir"], 1e-4, 1), curves["dir"])
    ax[1].set(title=f"Open-set identification (test ids) - {name}", xlabel="FPIR (log)", ylabel="DIR", xlim=(1e-4, 1), ylim=(0, 1.005))
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(VIS_DIR / f"recognition_{name}.png", dpi=120)
    plt.close(fig)


def update_profile(embedder: str, calibration_path: str) -> None:
    """Point configs/recognition.yaml -> matching.profiles.<embedder>.calibration at the new file."""
    p = CONFIG_DIR / "recognition.yaml"
    text = p.read_text()
    new, n = re.subn(rf"(\n\s+{embedder}:\s*\n\s+calibration:\s*)\S+", rf"\g<1>{calibration_path}", text, count=1)
    if n != 1:
        raise RuntimeError(f"could not find matching.profiles.{embedder}.calibration in {p}")
    p.write_text(new)


def lfw_quality(det_name: str, faces: dict, rel_paths: list[str], cfg: QualityConfig) -> dict[str, FaceQuality]:
    """Face quality per LFW image (cached); ``acceptable`` is re-derived from the current config."""
    cache = PROCESSED_DIR / "lfw" / f"quality_{det_name}.json"
    stored = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [r for r in rel_paths if r not in stored]
    root = RAW_DIR / "lfw" / "lfw"
    for r in tqdm(missing, desc="lfw quality", mininterval=2.0):
        f = faces[r]
        kps = np.array(f["keypoints"], np.float32)
        bbox = f.get("bbox") or [0, 0, 250, 250]
        det = FaceDetection(BoundingBox(*bbox), float(f.get("score", 0.0)), kps)
        stored[r] = assess_quality(det, align_face(load_image(root / r), kps), cfg).to_dict()
    if missing:
        cache.write_text(json.dumps(stored))
    out = {}
    for r in rel_paths:
        q = FaceQuality(**stored[r])
        q.acceptable = q.score >= cfg.min_quality
        out[r] = q
    return out


def decision_counts(results, truth: list[str] | None) -> dict[str, float]:
    """Fractions of match-correct / match-wrong / unknown / uncertain (truth None = unknown probes)."""
    n = max(len(results), 1)
    if truth is None:
        return {"false_match": sum(r.decision == MATCH for r in results) / n,
                "unknown": sum(r.decision == NO_MATCH for r in results) / n,
                "uncertain": sum(r.decision == UNCERTAIN for r in results) / n, "n": len(results)}
    return {"correct_match": sum(r.decision == MATCH and r.person_id == t for r, t in zip(results, truth)) / n,
            "wrong_person": sum(r.decision == MATCH and r.person_id != t for r, t in zip(results, truth)) / n,
            "missed_as_unknown": sum(r.decision == NO_MATCH for r in results) / n,
            "uncertain": sum(r.decision == UNCERTAIN for r in results) / n, "n": len(results)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedders", nargs="*")
    ap.add_argument("--alignment-detector")
    ap.add_argument("--flip", choices=["config", "on", "off"], default="config",
                    help="horizontal-flip test-time augmentation (default: as in configs/recognition.yaml)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _flip_scrfd")
    ap.add_argument("--update-config", action="store_true",
                    help="point configs/recognition.yaml at the new calibration file for each embedder")
    ap.add_argument("--verify-far", type=float, default=1e-4, help="1:1 false-accept target for the verify threshold")
    ap.add_argument("--target-fnir", type=float, default=0.01,
                    help="genuine-reject target defining the lower (reject) threshold of the uncertain band")
    args = ap.parse_args()
    log = setup_logging(log_file=resolve_path("outputs/reports/evaluate_recognition.log"))
    ensure_dirs()
    ecfg = load_config("evaluation")
    rc = ecfg["recognition"]
    rcfg = load_config("recognition")
    det_name = args.alignment_detector or rc["alignment_detector"]
    faces_doc = json.loads((PROCESSED_DIR / "lfw" / f"faces_{det_name}.json").read_text())
    faces = faces_doc["faces"]
    pairs = json.loads(resolve_path(rc["pairs"]).read_text())
    openset = json.loads(resolve_path(rc["openset"]).read_text())

    flat_pairs = [(p["a"], p["b"], p["same"], k) for k, fold in enumerate(pairs["folds"]) for p in fold]
    needed = {x for a, b, *_ in flat_pairs for x in (a, b)}
    for part in ("calibration", "test"):
        needed |= {x for imgs in openset[part]["gallery"].values() for x in imgs}
        needed |= {p["image"] for p in openset[part]["known_probes"] + openset[part]["unknown_probes"]}
    needed_l = sorted(needed)
    fallback = sorted(r for r in needed_l if faces[r]["status"].startswith("fallback"))

    for name in args.embedders or rc["embedders"]:
        t0 = time.perf_counter()
        ecfg_name = dict(rcfg[name])
        if args.flip != "config":
            ecfg_name["flip_augment"] = args.flip == "on"
        embedder = create_embedder({**rcfg, "backend": name, name: ecfg_name})
        variant = f"{name}{'_flip' if embedder.flip_augment else ''}_{det_name}"
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        with PeakRSS() as mem:
            emb = embed_images(embedder, faces, needed_l, PROCESSED_DIR / "lfw" / f"embeddings_{variant}.npz")
        embed_s = time.perf_counter() - t0

        # --- verification -------------------------------------------------------------------
        a = np.stack([emb[p[0]] for p in flat_pairs])
        b = np.stack([emb[p[1]] for p in flat_pairs])
        scores = (a * b).sum(1)
        labels = np.array([p[2] for p in flat_pairs])
        folds = np.array([p[3] for p in flat_pairs])
        ver_roc = roc(scores, labels)
        verification = {
            "protocol": "LFW View 2 (6000 pairs, 10 folds), unrestricted with labeled outside data",
            "auc": ver_roc["auc"], "eer": eer(scores, labels),
            "tar_at_far": tar_at_far(scores, labels, rc["far_targets"]),
            "tar_at_far_note": "thresholds chosen on the same 6000 pairs (descriptive ROC points, not held-out)",
            "ten_fold": kfold_accuracy(scores, labels, folds),
            "genuine_score_mean": float(scores[labels].mean()), "impostor_score_mean": float(scores[~labels].mean()),
        }

        # --- open-set identification ----------------------------------------------------------
        ident: dict = {"protocol": "identity-disjoint calibration/test halves of LFW; 2 enrollment images per known identity; "
                                   "known probes exclude enrollment images",
                       "split_stats": openset["stats"]}
        curves = None
        for strategy in ("mean", "max"):
            res: dict = {}
            sims = {}
            for part in ("calibration", "test"):
                spec = openset[part]
                g = build_gallery(name, embedder.dim, spec["gallery"], emb)
                kp = np.stack([emb[p["image"]] for p in spec["known_probes"]])
                up = np.stack([emb[p["image"]] for p in spec["unknown_probes"]])
                ids, ks = score_against_gallery(kp, g, strategy)
                _, us = score_against_gallery(up, g, strategy)
                sims[part] = (ids, ks, [p["identity"] for p in spec["known_probes"]], us)
            ids_c, ks_c, kl_c, us_c = sims["calibration"]
            ids_t, ks_t, kl_t, us_t = sims["test"]
            _, unk_top_c = top1(us_c, ids_c)
            for target in rc["calibration_fpir"]:
                thr = threshold_for_fpir(unk_top_c, target)
                res[f"fpir_{target}"] = {
                    "threshold_from_calibration": thr,
                    "calibration": open_set_metrics(ks_c, kl_c, us_c, ids_c, thr),
                    "test": open_set_metrics(ks_t, kl_t, us_t, ids_t, thr),
                }
            ident[strategy] = res
            if strategy == rc["strategy"]:
                curves = dir_fpir_curve(ks_t, kl_t, us_t, ids_t)
        op = ident[rc["strategy"]][f"fpir_{rc['operating_fpir']}"]
        op_thr = op["threshold_from_calibration"]
        verification["at_operating_threshold"] = far_frr(scores, labels, op_thr)

        # --- calibration (calibration identities only) + full decision rule on test identities ----
        qcfg = QualityConfig.from_dict(rcfg.get("quality"))
        quality = lfw_quality(det_name, faces, needed_l, qcfg)
        parts = {}
        for part in ("calibration", "test"):
            spec = openset[part]
            g = build_gallery(name, embedder.dim, spec["gallery"], emb)
            ids, templates = g.templates()
            col = {pid: i for i, pid in enumerate(ids)}
            kp_rel = [p["image"] for p in spec["known_probes"]]
            kl = [p["identity"] for p in spec["known_probes"]]
            up_rel = [p["image"] for p in spec["unknown_probes"]]
            ks = np.stack([emb[r] for r in kp_rel]) @ templates.T
            us = np.stack([emb[r] for r in up_rel]) @ templates.T
            own = np.zeros_like(ks, dtype=bool)
            own[np.arange(len(kl)), [col[x] for x in kl]] = True
            parts[part] = {"g": g, "kp": kp_rel, "kl": kl, "up": up_rel, "genuine": ks[own],
                           "impostor": np.concatenate([us.ravel(), ks[~own]]),
                           "kq": [quality[r] for r in kp_rel], "uq": [quality[r] for r in up_rel]}
        c, t_ = parts["calibration"], parts["test"]
        id_reject = min(op_thr, float(np.quantile(c["genuine"], args.target_fnir)))
        cal = ScoreCalibration.build(
            name, variant, c["impostor"], c["genuine"],
            source=f"LFW open-set calibration identities (seed {openset['seed']}), {det_name} alignment, "
                   f"flip={embedder.flip_augment}, strategy=mean",
        )
        v_accept = cal.threshold_at_far(args.verify_far)
        cal.identification = {"accept": op_thr, "reject": id_reject, "target_fpir": rc["operating_fpir"],
                              "target_fnir": args.target_fnir}
        cal.verification = {"accept": v_accept, "reject": min(v_accept, float(np.quantile(c["genuine"], args.target_fnir))),
                            "target_far": args.verify_far, "target_frr": args.target_fnir}
        cal_path = CONFIG_DIR / "calibration" / f"{variant}.json"
        cal.save(cal_path)

        margin = float(rcfg.get("matching", {}).get("margin", 0.0))
        decisions = {}
        for label, gate, mg in (("no_quality_gate_no_margin", False, 0.0), ("quality_gate", True, 0.0),
                                (f"quality_gate_margin_{margin}", True, margin)):
            m = Matcher(op_thr, "mean", 3, id_reject, mg, v_accept, cal.verification["reject"], cal)
            kr = m.identify_batch(np.stack([emb[r] for r in t_["kp"]]), t_["g"], t_["kq"] if gate else None)
            ur = m.identify_batch(np.stack([emb[r] for r in t_["up"]]), t_["g"], t_["uq"] if gate else None)
            decisions[label] = {"known_probes": decision_counts(kr, t_["kl"]), "unknown_probes": decision_counts(ur, None)}
        # 1:1 verification on test identities at the calibrated thresholds (all probe-template pairs)
        acc_q = np.array([q.acceptable for q in t_["kq"]])
        gen, imp = t_["genuine"], t_["impostor"]
        vr = cal.verification
        verify_test = {
            "genuine_pairs": int(len(gen)), "impostor_pairs": int(len(imp)),
            "accept_threshold": vr["accept"], "reject_threshold": vr["reject"],
            "far": float((imp >= vr["accept"]).mean()),
            "frr": float((gen < vr["reject"]).mean()),
            "genuine_uncertain": float(((gen >= vr["reject"]) & (gen < vr["accept"])).mean()),
            "impostor_uncertain": float(((imp >= vr["reject"]) & (imp < vr["accept"])).mean()),
            "frr_quality_ok_probes": float((gen[acc_q] < vr["reject"]).mean()) if acc_q.any() else None,
            "frr_low_quality_probes": float((gen[~acc_q] < vr["reject"]).mean()) if (~acc_q).any() else None,
            "low_quality_probe_fraction": float((~acc_q).mean()),
        }
        # Does the quality gate catch the hard cases? Error rate of LFW pairs at the EER threshold.
        thr_eer = verification["eer"]["threshold"]
        pair_ok = np.array([quality[p[0]].acceptable and quality[p[1]].acceptable for p in flat_pairs])
        err = (scores >= thr_eer) != labels
        quality_effect = {"threshold": thr_eer, "pairs_both_acceptable": int(pair_ok.sum()),
                          "error_rate_both_acceptable": float(err[pair_ok].mean()),
                          "pairs_with_low_quality": int((~pair_ok).sum()),
                          "error_rate_with_low_quality": float(err[~pair_ok].mean()) if (~pair_ok).any() else None}
        calibrated = {"file": str(cal_path.relative_to(CONFIG_DIR.parent)), "identification": cal.identification,
                      "verification": cal.verification, "n_impostor": cal.n_impostor, "n_genuine": cal.n_genuine,
                      "test_decisions": decisions, "test_verification_1to1": verify_test,
                      "quality_effect_lfw_pairs": quality_effect, "quality_config": vars(qcfg)}

        # --- performance ----------------------------------------------------------------------
        root = RAW_DIR / "lfw" / "lfw"
        sample = [align_face(load_image(root / r), np.array(faces[r]["keypoints"], np.float32)) for r in needed_l[:210]]
        perf = benchmark(lambda x: embedder.embed([x]), sample, warmup=ecfg["benchmark"]["warmup"])
        batch_perf = benchmark(lambda xs: embedder.embed(xs), [sample[i:i + 32] for i in range(0, 192, 32)], warmup=1)
        result = {
            "embedder": name, "alignment_detector": det_name, "flip_augment": embedder.flip_augment, "embedding_dim": embedder.dim,
            "images_embedded": len(needed_l),
            "alignment_fallback_images": fallback,
            "verification": verification,
            "identification": ident,
            "calibrated_decision": calibrated,
            "operating_threshold": {"value": op_thr, "strategy": rc["strategy"], "target_fpir": rc["operating_fpir"],
                                    "chosen_on": "calibration identities only"},
            "performance": {
                "model_load_seconds": load_s,
                "model_size_mb": model_size_mb([resolve_path(rcfg[name]["model_path"])]),
                "embedding_pass_seconds": embed_s, "embedding_pass_peak_rss_mb": mem.peak_mb,
                "latency_batch1_per_face": perf,
                "latency_batch32_per_batch": batch_perf,
                "input": "112x112 aligned face; timing excludes detection/alignment",
            },
            "hardware": hardware_info(), "date": datetime.now(timezone.utc).isoformat(),
        }
        (METRICS_DIR / f"recognition_{name}{args.tag}.json").write_text(json.dumps(result, indent=1))
        plots(f"{name}{args.tag}", result, ver_roc, curves)
        t = op["test"]
        log.info("%s: LFW 10-fold acc=%.4f±%.4f AUC=%.5f EER=%.4f | TAR@FAR=1e-3: %.4f | open-set test @thr=%.3f: "
                 "rank1=%.4f DIR=%.4f FPIR=%.4f unknown-reject=%.4f misID=%.4f | %.2f ms/face",
                 name, verification["ten_fold"]["mean_accuracy"], verification["ten_fold"]["std_accuracy"], ver_roc["auc"],
                 verification["eer"]["eer"], verification["tar_at_far"][0]["tar"], op_thr, t["rank1_accuracy_closed_set"],
                 t["dir"], t["fpir"], t["unknown_rejection_rate"], t["misidentification_rate"], perf["p50_ms"])
        d = decisions[f"quality_gate_margin_{margin}"]
        log.info("%s %s: id accept %.3f reject %.3f | verify accept %.3f (FAR %.0e) | test 1:N known: correct %.4f wrong %.4f "
                 "missed %.4f uncertain %.4f; strangers: false-match %.4f uncertain %.4f | test 1:1 FAR %.2e FRR %.4f",
                 name, variant, op_thr, id_reject, v_accept, args.verify_far, d["known_probes"]["correct_match"],
                 d["known_probes"]["wrong_person"], d["known_probes"]["missed_as_unknown"], d["known_probes"]["uncertain"],
                 d["unknown_probes"]["false_match"], d["unknown_probes"]["uncertain"], verify_test["far"], verify_test["frr"])
        if args.update_config:
            update_profile(name, calibrated["file"])
            log.info("configs/recognition.yaml: matching.profiles.%s.calibration = %s", name, calibrated["file"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
