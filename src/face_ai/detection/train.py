"""Transfer learning of a torchvision detector on WIDER FACE, with checkpoint save/resume.

The model is a COCO-pretrained ``fasterrcnn_mobilenet_v3_large_fpn`` whose anchors are shrunk for
faces and whose box predictor is replaced by a 2-class (background/face) head. This is a bounded,
reproducible experiment sized for a laptop GPU; it is not expected to beat purpose-built face
detectors trained for hundreds of epochs, and the default service detector remains YuNet unless
the evaluation says otherwise.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_ai.api.schemas import BoundingBox, FaceDetection
from face_ai.config import resolve_path
from face_ai.utils.image_io import ensure_bgr, load_image

log = logging.getLogger(__name__)


def build_model(arch: str = "fasterrcnn_mobilenet_v3_large_fpn", weights: str | None = "DEFAULT",
                min_size: int = 640, max_size: int = 1024):
    import torchvision
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if arch != "fasterrcnn_mobilenet_v3_large_fpn":
        raise ValueError(f"unsupported arch {arch}")
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights if weights not in (None, "none", "None") else None,
        min_size=min_size, max_size=max_size,
        box_detections_per_img=500, box_score_thresh=0.01,
        rpn_pre_nms_top_n_test=2000, rpn_post_nms_top_n_test=1000, box_batch_size_per_image=256,
    )
    # Same number of anchors per location (5 sizes x 3 ratios) so the pretrained RPN head is reused,
    # but sizes start at 8 px and ratios suit faces (taller than wide).
    sizes = ((8, 16, 32, 64, 128),) * 3
    ratios = ((1.0, 1.25, 1.5),) * 3
    model.rpn.anchor_generator = AnchorGenerator(sizes, ratios)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


class WiderDetectionDataset:
    """Images listed in a split file + boxes from the converted COCO JSON.

    Boxes flagged by preprocessing (degenerate / WIDER-invalid) and boxes smaller than
    ``min_face_px`` are excluded from training targets; the counts are recorded in ``stats``.
    """

    def __init__(self, split_file: str | Path, min_face_px: float = 8, max_images: int | None = None,
                 max_side: int = 1024, train: bool = True, seed: int = 42):
        split = json.loads(resolve_path(split_file).read_text())
        coco = json.loads(Path(split["coco"]).read_text())
        self.root = Path(split["image_root"])
        by_name = {im["file_name"]: im for im in coco["images"]}
        anns: dict[int, list] = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        names = list(split["images"])
        if max_images and len(names) > max_images:
            names = sorted(random.Random(seed).sample(names, max_images))
        self.items = []
        self.stats = {"images": 0, "boxes_used": 0, "boxes_flagged_ignore": 0, "boxes_below_min_size": 0}
        for n in names:
            boxes = []
            for a in anns.get(by_name[n]["id"], []):
                x, y, w, h = a["bbox"]
                if a["iscrowd"]:
                    self.stats["boxes_flagged_ignore"] += 1
                elif min(w, h) < min_face_px:
                    self.stats["boxes_below_min_size"] += 1
                else:
                    boxes.append([x, y, x + w, y + h])
            self.items.append((n, np.array(boxes, np.float32).reshape(-1, 4)))
            self.stats["boxes_used"] += len(boxes)
        self.stats["images"] = len(self.items)
        self.max_side = max_side
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        import torch

        name, boxes = self.items[i]
        img = cv2.cvtColor(load_image(self.root / name), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes = boxes.copy()
        if self.train and random.random() < 0.5:
            img = img[:, ::-1]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        boxes = boxes[keep]
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        target = {"boxes": torch.from_numpy(boxes).float(), "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return t, target, name


def _collate(batch):
    return tuple(zip(*batch))


def simple_ap(preds: dict[str, np.ndarray], gts: dict[str, np.ndarray], iou: float = 0.5) -> float:
    """VOC all-point AP over all target faces (used only for checkpoint selection on val_ft)."""
    from face_ai.evaluation.detection_metrics import bbox_overlaps, voc_ap

    scores, tps, n_gt = [], [], 0
    for k, gt in gts.items():
        n_gt += len(gt)
        p = preds.get(k, np.zeros((0, 5)))
        p = p[np.argsort(-p[:, 4])]
        matched = np.zeros(len(gt), bool)
        ov = bbox_overlaps(p[:, :4], gt) if len(gt) and len(p) else np.zeros((len(p), len(gt)))
        for j in range(len(p)):
            scores.append(p[j, 4])
            if ov.shape[1]:
                m = int(ov[j].argmax())
                if ov[j, m] >= iou and not matched[m]:
                    matched[m] = True
                    tps.append(1)
                    continue
            tps.append(0)
    if not scores:
        return 0.0
    order = np.argsort(-np.array(scores))
    tp = np.cumsum(np.array(tps)[order])
    fp = np.cumsum(1 - np.array(tps)[order])
    return voc_ap(tp / max(n_gt, 1), tp / np.maximum(tp + fp, 1e-9))


def evaluate_model(model, dataset: WiderDetectionDataset, device, batch_size: int = 4) -> float:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    preds, gts = {}, {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=_collate)
    with torch.no_grad():
        for imgs, targets, names in loader:
            outs = model([im.to(device) for im in imgs])
            for o, t, n in zip(outs, targets, names):
                b = o["boxes"].cpu().numpy()
                preds[n] = np.column_stack([b, o["scores"].cpu().numpy()]) if len(b) else np.zeros((0, 5))
                gts[n] = t["boxes"].numpy()
    model.train()
    return simple_ap(preds, gts)


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, step: int, best: float, cfg: dict) -> None:
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step, "best_val_ap": best,
                "config": cfg, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state()}, tmp)
    tmp.replace(path)


def train(cfg: dict[str, Any], resume: str | Path | None = None, max_minutes: float | None = None) -> dict[str, Any]:
    """Run (or resume) fine-tuning. Returns a summary with per-epoch metrics and timings."""
    import torch
    from torch.utils.data import DataLoader

    from face_ai.utils.device import torch_device

    _set_seed(cfg["seed"])
    device = torch_device(cfg.get("device", "auto"))
    ckpt_dir = resolve_path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_ds = WiderDetectionDataset(cfg["train_split"], cfg["min_face_px"], cfg["max_train_images"],
                                     cfg["image_max_side"], train=True, seed=cfg["seed"])
    val_ds = WiderDetectionDataset(cfg["val_split"], cfg["min_face_px"], cfg["max_val_images"],
                                   cfg["image_max_side"], train=False, seed=cfg["seed"])
    model = build_model(cfg["arch"], cfg.get("weights"), min_size=cfg.get("image_min_side", 640),
                        max_size=cfg["image_max_side"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                        collate_fn=_collate, generator=g, persistent_workers=cfg["num_workers"] > 0)
    total_steps = cfg["epochs"] * len(loader)
    warm = cfg["warmup_iters"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))

    start_epoch, step, best = 0, 0, -1.0
    history: list[dict[str, Any]] = []
    hist_file = ckpt_dir / "history.json"
    if resume:
        ck = torch.load(resolve_path(resume), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, step, best = ck["epoch"] + 1, ck["step"], ck["best_val_ap"]
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"])
        if hist_file.exists():
            history = json.loads(hist_file.read_text())
        log.info("resumed from %s at epoch %d (step %d, best val AP %.4f)", resume, start_epoch, step, best)

    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    log.info("device=%s train=%s val=%s steps/epoch=%d", device, train_ds.stats, val_ds.stats, len(loader))
    t_start = time.time()
    model.train()
    stopped_early = False
    for epoch in range(start_epoch, cfg["epochs"]):
        t_ep = time.time()
        running: list[float] = []
        for imgs, targets, _ in loader:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(imgs, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: { {k: float(v) for k, v in losses.items()} }")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()
            scheduler.step()
            step += 1
            running.append(loss.item())
            if step % cfg["log_every"] == 0:
                log.info("epoch %d step %d loss %.4f lr %.5f (%.1f min elapsed)", epoch, step, np.mean(running[-cfg["log_every"]:]),
                         optimizer.param_groups[0]["lr"], (time.time() - t_start) / 60)
            if max_minutes and (time.time() - t_start) / 60 > max_minutes:
                stopped_early = True
                break
        val_ap = evaluate_model(model, val_ds, device, cfg["batch_size"])
        rec = {"epoch": epoch, "step": step, "train_loss": float(np.mean(running)) if running else None,
               "val_ft_ap": val_ap, "epoch_minutes": (time.time() - t_ep) / 60, "partial_epoch": stopped_early}
        history.append(rec)
        hist_file.write_text(json.dumps(history, indent=2))
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, step, max(best, val_ap), cfg)
        if val_ap > best:
            best = val_ap
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, step, best, cfg)
        log.info("epoch %d done: loss %.4f val_ft AP %.4f (best %.4f) in %.1f min", epoch, rec["train_loss"] or float("nan"),
                 val_ap, best, rec["epoch_minutes"])
        if stopped_early:
            log.warning("stopped after %.1f min (max_minutes); resume with --resume %s", max_minutes, ckpt_dir / "last.pt")
            break
    return {"device": str(device), "train_stats": train_ds.stats, "val_stats": val_ds.stats, "history": history,
            "best_val_ft_ap": best, "total_minutes": (time.time() - t_start) / 60, "stopped_early": stopped_early,
            "checkpoint_dir": str(ckpt_dir)}


class TorchvisionFaceDetector:
    """Inference wrapper for a checkpoint produced by :func:`train` (boxes only, no keypoints)."""

    name = "torchvision"

    def __init__(self, checkpoint: str | Path, score_threshold: float = 0.5, device: str = "auto"):
        import torch

        from face_ai.utils.device import torch_device

        ck = torch.load(resolve_path(checkpoint), map_location="cpu", weights_only=False)
        cfg = ck["config"]
        self.device = torch_device(device)
        self.model = build_model(cfg["arch"], None, min_size=cfg.get("image_min_side", 640), max_size=cfg["image_max_side"])
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.score_threshold = score_threshold

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        import torch

        rgb = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.no_grad():
            o = self.model([t])[0]
        boxes, scores = o["boxes"].cpu().numpy(), o["scores"].cpu().numpy()
        return [FaceDetection(BoundingBox(*map(float, b)), float(s), None, detector=self.name)
                for b, s in zip(boxes, scores) if s >= self.score_threshold]
