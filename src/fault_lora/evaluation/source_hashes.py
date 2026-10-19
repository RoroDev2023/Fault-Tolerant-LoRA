"""The one rule for comparing recorded source hashes with the current files.

A recorded hash is accepted only if it equals the current file's hash, or if the
exact (path, recorded, current) change is listed in a committed decision file:

- configs/accepted_dependency_changes.json: only requirements.txt and the lockfile.
- configs/accepted_source_changes.json: only Python files under src/fault_lora/.

Every accepted mismatch is returned so callers can write it into their records.
Anything else fails. Missing decision files mean no accepted changes.
"""

import json
from pathlib import Path
import re

from fault_lora.evaluation.clustered_baseline import sha256_file
from fault_lora.evaluation.fault_checks import require


DEPENDENCY_CHANGES = "configs/accepted_dependency_changes.json"
SOURCE_CHANGES = "configs/accepted_source_changes.json"
DEPENDENCY_FILES = ("requirements.txt", "requirements-macos-arm64.lock.txt")
DEPENDENCY_FIELDS = {"path", "old_sha256", "new_sha256", "reason", "date"}
SOURCE_FIELDS = DEPENDENCY_FIELDS | {"commit"}
DIGEST = re.compile(r"[0-9a-f]{64}")


def _validate_entries(entries, fields, allowed_path, label):
    if not isinstance(entries, list):
        raise ValueError(f"{label} must be a list of accepted changes")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"{label}: each entry must have exactly {sorted(fields)}")
        if not isinstance(entry["path"], str) or not allowed_path(entry["path"]):
            raise ValueError(f"{label}: path is not allowed in this list: {entry['path']!r}")
        if any(not isinstance(entry[key], str) or not DIGEST.fullmatch(entry[key]) for key in ("old_sha256", "new_sha256")) or entry["old_sha256"] == entry["new_sha256"]:
            raise ValueError(f"{label}: hashes must be two different lowercase SHA-256 digests: {entry['path']}")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise ValueError(f"{label}: a reason is required: {entry['path']}")
        if not isinstance(entry["date"], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry["date"]):
            raise ValueError(f"{label}: date must be YYYY-MM-DD: {entry['path']}")
        if "commit" in fields and entry["commit"] is not None and (not isinstance(entry["commit"], str) or not re.fullmatch(r"[0-9a-f]{7,40}", entry["commit"])):
            raise ValueError(f"{label}: commit must be null or a hexadecimal commit id: {entry['path']}")
        key = (entry["path"], entry["old_sha256"])
        if key in seen:
            raise ValueError(f"{label}: duplicate accepted change for {entry['path']}")
        seen.add(key)
    return entries


def is_dependency_file(path):
    return path in DEPENDENCY_FILES


def is_package_source(path):
    parts = Path(path).parts
    return path == Path(path).as_posix() and len(parts) > 2 and parts[:2] == ("src", "fault_lora") and path.endswith(".py") and ".." not in parts


def load_accepted_changes(root):
    """Read and validate both decision files; each entry is tagged with its list."""
    root = Path(root)
    accepted = []
    for relative, fields, allowed, kind in ((DEPENDENCY_CHANGES, DEPENDENCY_FIELDS, is_dependency_file, "dependency"),
                                            (SOURCE_CHANGES, SOURCE_FIELDS, is_package_source, "source")):
        path = root / relative
        if path.exists():
            entries = _validate_entries(json.loads(path.read_text()), fields, allowed, relative)
            accepted += [{**entry, "list": kind, "decision_file": relative} for entry in entries]
    return accepted


def compare_source_hashes(root, recorded, label, *, accepted=None):
    """Require every recorded file to match, or to be an exactly listed change.

    Returns the accepted mismatches (empty when everything matches exactly).
    """
    root = Path(root)
    accepted = load_accepted_changes(root) if accepted is None else accepted
    used = []
    for relative, digest in recorded.items():
        path = root / relative
        require(path.is_file(), f"{label}: {relative} (file is missing)")
        current = sha256_file(path)
        if current == digest:
            continue
        allowed = is_dependency_file if relative in DEPENDENCY_FILES else is_package_source
        matches = [entry for entry in accepted if entry["path"] == relative and entry["old_sha256"] == digest and entry["new_sha256"] == current
                   and (entry["list"] == "dependency") == is_dependency_file(relative) and allowed(relative)]
        require(len(matches) == 1, f"{label}: {relative}")
        entry = matches[0]
        used.append({"path": relative, "recorded_sha256": digest, "current_sha256": current, "list": entry["list"], "decision_file": entry["decision_file"],
                     "reason": entry["reason"], "date": entry["date"]})
    return used


def unique_changes(changes):
    """Collapse the accepted mismatches reported by many records into distinct changes."""
    distinct = {}
    for change in changes:
        distinct.setdefault((change["path"], change["recorded_sha256"], change["current_sha256"]), {**change, "records": 0})["records"] += 1
    return sorted(distinct.values(), key=lambda change: (change["list"], change["path"], change["recorded_sha256"]))


def parse_lockfile(text):
    """Map normalized package name to (version, sorted wheel hashes)."""
    packages = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;\\]+)((?:\s+--hash=sha256:[0-9a-f]{64})*)\s*\\?", line)
        require(match is not None, f"Unrecognized lockfile line: {line}")
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        require(name not in packages, f"Duplicate lockfile package: {name}")
        packages[name] = (match.group(2), sorted(re.findall(r"sha256:([0-9a-f]{64})", match.group(3))))
    return packages


def summarize_lockfile_change(old_text, new_text):
    old, new = parse_lockfile(old_text), parse_lockfile(new_text)
    kept = sorted(set(old) & set(new))
    return {"old_packages": len(old), "new_packages": len(new),
            "added": {name: new[name][0] for name in sorted(set(new) - set(old))},
            "removed": {name: old[name][0] for name in sorted(set(old) - set(new))},
            "version_changed": {name: {"old": old[name][0], "new": new[name][0]} for name in kept if old[name][0] != new[name][0]},
            "same_version": len([name for name in kept if old[name][0] == new[name][0]]),
            "same_version_with_different_wheel_hashes": [name for name in kept if old[name][0] == new[name][0] and old[name][1] != new[name][1]]}


def summarize_requirements_change(old_text, new_text):
    lines = lambda text: [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    old, new = lines(old_text), lines(new_text)
    return {"old_direct": old, "new_direct": new, "added": [line for line in new if line not in old], "removed": [line for line in old if line not in new]}
