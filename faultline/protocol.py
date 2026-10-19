"""Frozen benchmark protocol: one TOML file, loaded once and hashed.

The hash of the file's bytes is stamped on every certificate and result, so a run
can always be traced to the exact thresholds, seeds and severity grid it used.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "configs" / "protocol.toml"


@dataclass(frozen=True)
class Protocol:
    """The parsed protocol sections plus the hash of the source file.

    Args:
        values: The parsed TOML as nested dictionaries.
        hash: SHA-256 hex digest of the file's bytes.

    How:
        Read-only access by section: ``protocol["autospec"]["threshold"]``.
    """

    values: dict[str, Any]
    hash: str

    def __getitem__(self, section: str) -> Any:
        """Return one top-level section or value.

        Args:
            section: The TOML key.

        How:
            Delegates to the parsed dictionary.

        Returns:
            The section or value.
        """
        return self.values[section]


def load_protocol(path: Path | str = DEFAULT_PROTOCOL_PATH) -> Protocol:
    """Load and hash a protocol file.

    Args:
        path: Protocol TOML path; defaults to ``configs/protocol.toml``.

    How:
        Reads the bytes once, hashes them, and parses the same bytes, so the hash
        always matches the parsed content.

    Returns:
        The protocol.
    """
    raw = Path(path).read_bytes()
    return Protocol(values=tomllib.loads(raw.decode("utf-8")), hash=hashlib.sha256(raw).hexdigest())
