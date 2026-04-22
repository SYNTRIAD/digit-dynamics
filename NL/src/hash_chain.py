# Copyright (c) 2026 Remco Havenaar / SYNTRIAD Research — MIT License
"""
Standalone hash-chain audit infrastructure for digit-dynamics validation.

Provides an append-only, cryptographically chained audit log with:
  - SHA-256 content hashing
  - Chain integrity (each entry references previous hash)
  - HMAC-SHA256 session seal (tamper-evident certificate)
  - JSONL export for offline verification

No external dependencies beyond Python stdlib.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ChainEntry:
    """A single entry in the hash chain."""
    iteration: int
    timestamp: str
    event_type: str
    data: Dict[str, Any]
    content_hash: str
    prev_hash: str
    entry_hash: str


@dataclass
class ChainSession:
    """An active hash-chain session."""
    session_id: str
    created: str
    genesis_hash: str
    entries: List[ChainEntry] = field(default_factory=list)
    sealed: bool = False
    seal_hash: Optional[str] = None


# Module-level session store
_sessions: Dict[str, ChainSession] = {}


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _content_hash(event_type: str, data: Dict[str, Any]) -> str:
    """Deterministic hash of entry content."""
    canonical = json.dumps(
        {"event_type": event_type, "data": data},
        sort_keys=True, separators=(",", ":"),
    )
    return _sha256(canonical)


def _entry_hash(content_hash: str, prev_hash: str) -> str:
    """Chain hash: H(content_hash || prev_hash)."""
    return _sha256(content_hash + prev_hash)


def chain_init(session_id: Optional[str] = None) -> Dict[str, str]:
    """Initialize a new hash-chain session.

    Returns:
        {"session_id": ..., "genesis_hash": ...}
    """
    if session_id is None:
        session_id = f"val-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"

    genesis_data = {
        "session_id": session_id,
        "created": datetime.now().isoformat(),
        "system": "digit-dynamics-validation",
    }
    genesis_hash = _sha256(json.dumps(genesis_data, sort_keys=True))

    session = ChainSession(
        session_id=session_id,
        created=genesis_data["created"],
        genesis_hash=genesis_hash,
    )
    _sessions[session_id] = session

    return {"session_id": session_id, "genesis_hash": genesis_hash}


def chain_log(
    session_id: str,
    event_type: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Append an entry to the hash chain.

    Returns:
        {"iteration": N, "hash": "...", "prev_hash": "..."}
    """
    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(f"Unknown session: {session_id}")
    if session.sealed:
        raise RuntimeError(f"Session {session_id} is already sealed")

    prev = session.entries[-1].entry_hash if session.entries else session.genesis_hash
    iteration = len(session.entries) + 1
    ts = datetime.now().isoformat()

    c_hash = _content_hash(event_type, data)
    e_hash = _entry_hash(c_hash, prev)

    entry = ChainEntry(
        iteration=iteration,
        timestamp=ts,
        event_type=event_type,
        data=data,
        content_hash=c_hash,
        prev_hash=prev,
        entry_hash=e_hash,
    )
    session.entries.append(entry)

    return {"iteration": iteration, "hash": e_hash, "prev_hash": prev}


def chain_seal(session_id: str) -> Dict[str, Any]:
    """Seal the session with HMAC-SHA256 certificate.

    The HMAC key is derived from the genesis hash (self-referential integrity).

    Returns:
        Full certificate dict.
    """
    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(f"Unknown session: {session_id}")
    if session.sealed:
        raise RuntimeError(f"Session {session_id} is already sealed")

    # Build chain summary for HMAC
    chain_data = {
        "session_id": session.session_id,
        "genesis_hash": session.genesis_hash,
        "num_entries": len(session.entries),
        "final_hash": session.entries[-1].entry_hash if session.entries else session.genesis_hash,
        "created": session.created,
        "sealed": datetime.now().isoformat(),
    }
    chain_json = json.dumps(chain_data, sort_keys=True, separators=(",", ":"))

    # HMAC with genesis hash as key
    seal = hmac.new(
        session.genesis_hash.encode("utf-8"),
        chain_json.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    session.sealed = True
    session.seal_hash = seal

    certificate = {
        **chain_data,
        "hmac_sha256": seal,
        "algorithm": "HMAC-SHA256(genesis_hash, chain_summary)",
    }
    return certificate


def chain_verify(certificate: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a certificate's HMAC integrity.

    Returns:
        {"valid": bool, "details": str}
    """
    required = ["session_id", "genesis_hash", "num_entries", "final_hash",
                 "created", "sealed", "hmac_sha256"]
    for key in required:
        if key not in certificate:
            return {"valid": False, "details": f"Missing field: {key}"}

    chain_data = {
        "session_id": certificate["session_id"],
        "genesis_hash": certificate["genesis_hash"],
        "num_entries": certificate["num_entries"],
        "final_hash": certificate["final_hash"],
        "created": certificate["created"],
        "sealed": certificate["sealed"],
    }
    chain_json = json.dumps(chain_data, sort_keys=True, separators=(",", ":"))

    expected = hmac.new(
        certificate["genesis_hash"].encode("utf-8"),
        chain_json.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if hmac.compare_digest(expected, certificate["hmac_sha256"]):
        return {"valid": True, "details": "HMAC verification passed"}
    else:
        return {"valid": False, "details": "HMAC mismatch — certificate tampered"}


def chain_verify_log(entries_jsonl: str) -> Dict[str, Any]:
    """Verify the integrity of a JSONL audit log.

    Args:
        entries_jsonl: String with one JSON object per line.

    Returns:
        {"valid": bool, "num_entries": int, "details": str}
    """
    lines = [l.strip() for l in entries_jsonl.strip().split("\n") if l.strip()]
    if not lines:
        return {"valid": True, "num_entries": 0, "details": "Empty log"}

    prev_hash = None
    for i, line in enumerate(lines):
        entry = json.loads(line)
        if i == 0:
            # Genesis entry
            prev_hash = entry.get("genesis_hash") or entry.get("entry_hash")
            if "genesis_hash" in entry:
                continue

        expected_content = _content_hash(entry["event_type"], entry["data"])
        expected_entry = _entry_hash(expected_content, entry["prev_hash"])

        if entry["prev_hash"] != prev_hash:
            return {
                "valid": False,
                "num_entries": i,
                "details": f"Chain broken at entry {i}: prev_hash mismatch",
            }
        if entry["entry_hash"] != expected_entry:
            return {
                "valid": False,
                "num_entries": i,
                "details": f"Content tampered at entry {i}: entry_hash mismatch",
            }
        prev_hash = entry["entry_hash"]

    return {"valid": True, "num_entries": len(lines), "details": "Chain intact"}


def get_session(session_id: str) -> ChainSession:
    """Get session object (for export)."""
    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(f"Unknown session: {session_id}")
    return session


def export_jsonl(session_id: str) -> str:
    """Export session as JSONL string."""
    session = get_session(session_id)
    lines = []

    # Genesis line
    genesis = {
        "type": "genesis",
        "session_id": session.session_id,
        "genesis_hash": session.genesis_hash,
        "created": session.created,
    }
    lines.append(json.dumps(genesis, sort_keys=True))

    # Entry lines
    for entry in session.entries:
        line = {
            "iteration": entry.iteration,
            "timestamp": entry.timestamp,
            "event_type": entry.event_type,
            "data": entry.data,
            "content_hash": entry.content_hash,
            "prev_hash": entry.prev_hash,
            "entry_hash": entry.entry_hash,
        }
        lines.append(json.dumps(line, sort_keys=True))

    return "\n".join(lines) + "\n"
