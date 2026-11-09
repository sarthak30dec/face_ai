"""Enrollment gallery: per-person reference embeddings plus consent metadata.

Storage layout (``<dir>/``, created with 0700/0600 permissions)::

    gallery.json     model name, dim, per-person metadata, consent record and the optional
                     user-entered profile card (name, occupation, about, link); no raw images
    embeddings.npz   person_id -> (k, dim) float32 reference embeddings

Galleries built for benchmark evaluation are ``ephemeral`` and cannot be saved, so public-dataset
identities are never written into a product gallery.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from face_ai.recognition.embedder import l2_normalize

GALLERY_VERSION = 1


class GalleryError(RuntimeError):
    pass


@dataclass
class ConsentRecord:
    """Who agreed to enrollment, when, and for what purpose. Required for every person."""

    granted: bool
    granted_by: str  # the person themself or their legal guardian
    purpose: str
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).date().isoformat())
    retention_days: int | None = None
    notes: str = ""


PROFILE_FIELDS = {"display_name": 80, "occupation": 80, "about": 280, "link": 300}


def clean_profile(profile: dict | None) -> dict[str, str]:
    """Keep known fields, trim lengths, and only allow http(s) links."""
    out: dict[str, str] = {}
    for key, limit in PROFILE_FIELDS.items():
        v = str((profile or {}).get(key) or "").strip()[:limit]
        if key == "link" and v and not v.lower().startswith(("https://", "http://")):
            raise GalleryError("profile link must start with https://")
        if v:
            out[key] = v
    return out


@dataclass
class PersonRecord:
    person_id: str
    embeddings: np.ndarray  # (k, dim), L2-normalized
    consent: ConsentRecord | None
    created_at: str
    updated_at: str
    source_hashes: list[str] = field(default_factory=list)  # sha256 of source images; never paths/pixels
    profile: dict[str, str] = field(default_factory=dict)  # user-entered card: display_name, occupation, about, link

    @property
    def num_references(self) -> int:
        return int(self.embeddings.shape[0])


class Gallery:
    def __init__(self, model: str, dim: int, ephemeral: bool = False):
        self.model = model
        self.dim = dim
        self.ephemeral = ephemeral
        self._people: dict[str, PersonRecord] = {}

    # --- mutation ---------------------------------------------------------------------------
    def add(self, person_id: str, embeddings: np.ndarray, consent: ConsentRecord | None,
            source_hashes: list[str] | None = None) -> PersonRecord:
        embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if embeddings.shape[1] != self.dim:
            raise GalleryError(f"embedding dim {embeddings.shape[1]} != gallery dim {self.dim} ({self.model})")
        if not self.ephemeral and (consent is None or not consent.granted):
            raise GalleryError("enrollment requires a ConsentRecord with granted=True")
        now = datetime.now(timezone.utc).isoformat()
        embeddings = l2_normalize(embeddings)
        if person_id in self._people:
            rec = self._people[person_id]
            rec.embeddings = np.vstack([rec.embeddings, embeddings])
            rec.updated_at = now
            rec.source_hashes.extend(source_hashes or [])
            if consent is not None:
                rec.consent = consent
        else:
            rec = PersonRecord(person_id, embeddings, consent, now, now, list(source_hashes or []))
            self._people[person_id] = rec
        return rec

    def remove(self, person_id: str) -> bool:
        return self._people.pop(person_id, None) is not None

    def set_profile(self, person_id: str, profile: dict | None) -> dict[str, str]:
        """Replace the person's profile card with ``profile`` (validated by ``clean_profile``)."""
        rec = self._people[person_id]
        rec.profile = clean_profile(profile)
        rec.updated_at = datetime.now(timezone.utc).isoformat()
        return rec.profile

    # --- queries ------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._people)

    def __contains__(self, person_id: str) -> bool:
        return person_id in self._people

    def people(self) -> list[str]:
        return sorted(self._people)

    def get(self, person_id: str) -> PersonRecord:
        return self._people[person_id]

    def templates(self) -> tuple[list[str], np.ndarray]:
        """One template per person: the re-normalized mean of its reference embeddings."""
        ids = self.people()
        if not ids:
            return [], np.zeros((0, self.dim), np.float32)
        mat = np.stack([l2_normalize(self._people[i].embeddings.mean(0)) for i in ids])
        return ids, mat.astype(np.float32)

    def references(self) -> tuple[list[str], np.ndarray]:
        """All reference embeddings with their owner ids (for max-over-references matching)."""
        owners: list[str] = []
        mats = []
        for pid in self.people():
            e = self._people[pid].embeddings
            owners.extend([pid] * e.shape[0])
            mats.append(e)
        return owners, (np.vstack(mats) if mats else np.zeros((0, self.dim), np.float32))

    # --- persistence --------------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        if self.ephemeral:
            raise GalleryError("ephemeral (evaluation) galleries must not be persisted")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        meta = {
            "version": GALLERY_VERSION,
            "model": self.model,
            "dim": self.dim,
            "order": self.people(),  # row i of embeddings.npz key "p{i}"
            "people": {
                pid: {
                    "consent": asdict(r.consent) if r.consent else None,
                    "created_at": r.created_at,
                    "updated_at": r.updated_at,
                    "num_references": r.num_references,
                    "source_hashes": r.source_hashes,
                    "profile": r.profile,
                }
                for pid, r in self._people.items()
            },
        }
        tmp_json, tmp_npz = d / "gallery.json.tmp", d / "embeddings.tmp.npz"
        tmp_json.write_text(json.dumps(meta, indent=2))
        np.savez(tmp_npz, **{f"p{i}": self._people[pid].embeddings for i, pid in enumerate(self.people())})
        for tmp, final in ((tmp_json, d / "gallery.json"), (tmp_npz, d / "embeddings.npz")):
            os.chmod(tmp, 0o600)
            tmp.replace(final)
        return d

    @classmethod
    def load(cls, directory: str | Path, expected_model: str | None = None) -> "Gallery":
        d = Path(directory)
        meta = json.loads((d / "gallery.json").read_text())
        if expected_model and meta["model"] != expected_model:
            raise GalleryError(
                f"gallery was built with {meta['model']!r} but the active embedder is {expected_model!r}; "
                "embeddings from different models are not comparable. Re-enroll with the active model."
            )
        g = cls(meta["model"], int(meta["dim"]))
        with np.load(d / "embeddings.npz", allow_pickle=False) as npz:
            for i, pid in enumerate(meta["order"]):
                m = meta["people"][pid]
                consent = ConsentRecord(**m["consent"]) if m.get("consent") else None
                g._people[pid] = PersonRecord(pid, npz[f"p{i}"].astype(np.float32), consent, m["created_at"],
                                              m["updated_at"], list(m.get("source_hashes", [])),
                                              dict(m.get("profile") or {}))
        return g

    @staticmethod
    def delete(directory: str | Path) -> None:
        """Remove a stored gallery (files are unlinked; secure erase depends on the filesystem)."""
        d = Path(directory)
        for name in ("gallery.json", "embeddings.npz"):
            p = d / name
            if p.exists():
                p.unlink()
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
