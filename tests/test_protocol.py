"""The frozen protocol loads, hashes its own bytes, and keeps the agreed decisions."""

from __future__ import annotations

import hashlib

from faultline.protocol import DEFAULT_PROTOCOL_PATH, load_protocol


def test_protocol_hash_is_the_hash_of_the_file_bytes():
    protocol = load_protocol()
    assert protocol.hash == hashlib.sha256(DEFAULT_PROTOCOL_PATH.read_bytes()).hexdigest()


def test_protocol_records_the_agreed_decisions():
    protocol = load_protocol()
    assert protocol["autospec"]["threshold"] == 0.8
    assert protocol["certification"]["p_gold_min"] == 0.85
    assert protocol["autospec"]["method"] == "dirl"


def test_certification_and_evaluation_seeds_are_disjoint():
    seeds = load_protocol()["seeds"]
    assert not set(seeds["certification"]) & set(seeds["evaluation"])
