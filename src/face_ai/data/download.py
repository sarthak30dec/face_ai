"""Resumable, checksum-verified downloads with safe archive extraction.

Used for both datasets (``configs/datasets.yaml``) and pretrained models
(``configs/models.yaml``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import stat
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

USER_AGENT = "face_ai-dataset-downloader/0.1 (+research; python-requests)"
CHUNK = 1 << 20
EXTRACT_MARKER = ".extracted.json"


class DownloadError(RuntimeError):
    pass


class UnsafeArchiveError(RuntimeError):
    pass


@dataclass
class RemoteFile:
    filename: str
    url: str
    sha256: str | None = None
    size: int | None = None
    extract: bool = False
    optional: bool = False
    checksum_source: str | None = None
    source_note: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RemoteFile":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass
class FileResult:
    filename: str
    url: str
    path: str
    size: int
    sha256: str
    checksum_verified: bool
    checksum_source: str | None
    downloaded: bool  # False when an existing valid copy was reused
    extracted_to: str | None = None
    extracted_members: int | None = None
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: Path, chunk: int = CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def check_disk_space(directory: Path, required_bytes: int, margin_bytes: int = 1 << 30) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < required_bytes + margin_bytes:
        raise DownloadError(
            f"Not enough disk space in {directory}: need ~{human_bytes(required_bytes + margin_bytes)} "
            f"(including {human_bytes(margin_bytes)} safety margin), have {human_bytes(free)}. "
            "Free space or set FACE_AI_DATA_DIR to a larger volume."
        )


def _verify(path: Path, rf: RemoteFile) -> tuple[str, bool]:
    """Return (sha256, verified). Raises if a published checksum mismatches."""
    if rf.size is not None and path.stat().st_size != rf.size:
        raise DownloadError(f"{path.name}: size {path.stat().st_size} != expected {rf.size}")
    digest = sha256_file(path)
    if rf.sha256:
        if digest.lower() != rf.sha256.lower():
            raise DownloadError(f"{path.name}: sha256 {digest} != expected {rf.sha256}")
        return digest, True
    return digest, False


def download_file(
    rf: RemoteFile,
    dest_dir: Path,
    retries: int = 6,
    backoff: float = 2.0,
    timeout: float = 60.0,
    session: requests.Session | None = None,
) -> FileResult:
    """Download ``rf`` into ``dest_dir`` resuming from ``<file>.part`` when possible.

    An existing complete file is reused if it passes size/sha256 checks.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / rf.filename
    part = dest.with_name(dest.name + ".part")
    t0 = time.time()

    if dest.exists():
        try:
            digest, verified = _verify(dest, rf)
            log.info("%s: valid copy present (%s), skipping download", rf.filename, human_bytes(dest.stat().st_size))
            return FileResult(rf.filename, rf.url, str(dest), dest.stat().st_size, digest, verified,
                              rf.checksum_source if verified else "computed (no published checksum)", False,
                              seconds=time.time() - t0)
        except DownloadError as e:
            log.warning("%s: existing file invalid (%s); re-downloading", rf.filename, e)
            dest.unlink()

    if rf.size:
        have = part.stat().st_size if part.exists() else 0
        check_disk_space(dest_dir, rf.size - have)

    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", USER_AGENT)
    attempt = 0
    while True:
        attempt += 1
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with sess.get(rf.url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
                if r.status_code == 416:  # requested range beyond end: .part is already complete
                    break
                if r.status_code in (401, 403):
                    raise DownloadError(
                        f"{rf.filename}: HTTP {r.status_code} from {r.url.split('?')[0]}. The host requires "
                        "authentication or has restricted access; this is not bypassed. Obtain access "
                        "manually or choose an alternative source."
                    )
                if r.status_code == 404:
                    raise DownloadError(f"{rf.filename}: HTTP 404 at {rf.url} (endpoint moved or removed)")
                r.raise_for_status()
                if offset and r.status_code != 206:
                    log.warning("%s: server ignored Range request; restarting from 0", rf.filename)
                    offset = 0
                total = None
                if r.status_code == 206 and "Content-Range" in r.headers:
                    total = int(r.headers["Content-Range"].rsplit("/", 1)[-1])
                elif "Content-Length" in r.headers:
                    total = int(r.headers["Content-Length"]) + offset
                mode = "ab" if offset else "wb"
                with open(part, mode) as f, tqdm(
                    total=total, initial=offset, unit="B", unit_scale=True, unit_divisor=1024,
                    desc=rf.filename, mininterval=1.0, leave=True,
                ) as bar:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            break
        except DownloadError:
            raise
        except (requests.RequestException, OSError) as e:
            if attempt >= retries:
                raise DownloadError(
                    f"{rf.filename}: failed after {attempt} attempts ({e}). Partial data kept at {part}; "
                    "re-run the same command to resume."
                ) from e
            wait = backoff ** attempt
            log.warning("%s: attempt %d failed (%s); retrying in %.0fs", rf.filename, attempt, e, wait)
            time.sleep(wait)

    try:
        digest, verified = _verify(part, rf)
    except DownloadError:
        bad = part.with_name(part.name + ".corrupt")
        part.rename(bad)
        log.error("%s: integrity check failed; moved to %s. Re-run to download again.", rf.filename, bad)
        raise
    part.rename(dest)
    log.info("%s: downloaded %s, sha256 %s (%s)", rf.filename, human_bytes(dest.stat().st_size), digest[:16],
             "verified" if verified else "no published checksum; recorded")
    return FileResult(rf.filename, rf.url, str(dest), dest.stat().st_size, digest, verified,
                      rf.checksum_source if verified else "computed (no published checksum)", True,
                      seconds=time.time() - t0)


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract(archive: Path, out_dir: Path, archive_sha256: str | None = None) -> int:
    """Extract a zip/tar archive into ``out_dir``, refusing path traversal, absolute paths and links.

    Writes a marker so an unchanged archive is not extracted twice. Returns the member count.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f"{EXTRACT_MARKER[:-5]}.{archive.name}.json"
    if marker.exists() and archive_sha256:
        info = json.loads(marker.read_text())
        if info.get("sha256") == archive_sha256:
            log.info("%s: already extracted to %s", archive.name, out_dir)
            return int(info.get("members", 0))

    name = archive.name.lower()
    count = 0
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise UnsafeArchiveError(f"{archive.name}: CRC error in member {bad}")
            members = zf.infolist()
            for m in members:
                mode = (m.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise UnsafeArchiveError(f"{archive.name}: symlink member {m.filename} refused")
                if m.filename.startswith(("/", "\\")) or ".." in Path(m.filename).parts or \
                        not _is_within(out_dir, out_dir / m.filename):
                    raise UnsafeArchiveError(f"{archive.name}: unsafe path {m.filename!r}")
            for m in tqdm(members, desc=f"extract {archive.name}", unit="file", mininterval=1.0):
                zf.extract(m, out_dir)
                count += 1
    elif name.endswith((".tgz", ".tar.gz", ".tar", ".tar.bz2", ".tar.xz")):
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            for m in members:
                if m.issym() or m.islnk() or m.isdev():
                    raise UnsafeArchiveError(f"{archive.name}: link/device member {m.name} refused")
                if m.name.startswith("/") or ".." in Path(m.name).parts or not _is_within(out_dir, out_dir / m.name):
                    raise UnsafeArchiveError(f"{archive.name}: unsafe path {m.name!r}")
            for m in tqdm(members, desc=f"extract {archive.name}", unit="file", mininterval=1.0):
                tf.extract(m, out_dir, filter="data")
                count += 1
    else:
        raise ValueError(f"Unsupported archive type: {archive}")

    marker.write_text(json.dumps({
        "archive": archive.name, "sha256": archive_sha256, "members": count,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return count


def fetch_all(
    files: list[RemoteFile],
    archive_dir: Path,
    extract_dir: Path,
    include_optional: bool = False,
) -> tuple[list[FileResult], list[str]]:
    """Download (and extract) each file; collect errors instead of aborting on the first one."""
    selected = [f for f in files if include_optional or not f.optional]
    needed = sum(f.size or 0 for f in selected if not (archive_dir / f.filename).exists())
    needed += sum(f.size or 0 for f in selected if f.extract)  # extracted copy is roughly archive size
    check_disk_space(archive_dir, needed)

    results: list[FileResult] = []
    errors: list[str] = []
    for rf in selected:
        try:
            res = download_file(rf, archive_dir)
            if rf.extract:
                res.extracted_members = safe_extract(Path(res.path), extract_dir, res.sha256)
                res.extracted_to = str(extract_dir)
            results.append(res)
        except Exception as e:  # noqa: BLE001 - recorded and reported per file
            log.error("%s: %s", rf.filename, e)
            errors.append(f"{rf.filename}: {e}")
    return results, errors
