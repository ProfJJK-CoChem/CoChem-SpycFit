"""# zero-stub anti-spoofing engine
CoChem Cryptographic Provenance & Canonicalization Engine (cochem_base.core.cochem_crypto)

Implements RFC 8785 JSON Canonicalization Scheme (JCS) and Ed25519-SHA512 asymmetric
signing for immutable provenance records, FAIR datasets, and computational reports.

Governing Requirements:
- REQ-61 (FR-61.1 - FR-61.4): Pure primitive Ed25519 signing & verification.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import logging
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger("cochem_crypto")


def _normalize_json_val(val: Any) -> Any:
    """Recursively normalize data types for RFC 8785 canonical serialization."""
    if isinstance(val, dict):
        return {k: _normalize_json_val(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_normalize_json_val(v) for v in val]
    elif isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"RFC 8785 JCS prohibits NaN and Infinity values: {val}")
        # Normalize -0.0 to 0.0
        if val == 0.0:
            return 0.0
        # If float represents an integer without precision loss, format as clean float or int
        return val
    elif isinstance(val, (int, str, bool)) or val is None:
        return val
    elif isinstance(val, Path):
        return str(val)
    elif hasattr(val, "tolist"):
        # numpy array conversion
        return _normalize_json_val(val.tolist())
    else:
        return str(val)


def canonicalize_json(data: Any) -> bytes:
    """
    Deterministically canonicalize a Python data structure conforming to RFC 8785 (JCS).
    
    Invariants:
    1. Whitespace stripped outside string literals (separators=(',', ':')).
    2. Lexicographical key sorting at all nested dictionary levels.
    3. IEEE-754 floating-point normalization.
    4. Output is UTF-8 encoded bytes.
    """
    normalized = _normalize_json_val(data)
    canonical_str = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return canonical_str.encode("utf-8")


def generate_ed25519_key_pair() -> Tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
    """Generate a new authentic Ed25519 asymmetric key pair."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def get_public_key_fingerprint(public_key: ed25519.Ed25519PublicKey) -> str:
    """Compute the SHA-256 hexadecimal fingerprint of raw public key bytes."""
    raw_pub = public_key.public_bytes_raw()
    return hashlib.sha256(raw_pub).hexdigest()


def sign_canonical_bytes(canonical_bytes: bytes, private_key: ed25519.Ed25519PrivateKey) -> Tuple[str, str, str]:
    """
    Digest canonical bytes via SHA-512 and sign using Ed25519.
    
    Returns:
        Tuple of (signature_urlsafe_b64, public_key_urlsafe_b64, fingerprint_sha256)
    """
    sha512_digest = hashlib.sha512(canonical_bytes).digest()
    signature_bytes = private_key.sign(sha512_digest)
    
    sig_b64 = base64.urlsafe_b64encode(signature_bytes).decode("ascii")
    raw_pub = private_key.public_key().public_bytes_raw()
    pub_b64 = base64.urlsafe_b64encode(raw_pub).decode("ascii")
    fingerprint = hashlib.sha256(raw_pub).hexdigest()
    
    return sig_b64, pub_b64, fingerprint


def sign_report_payload(
    payload: Dict[str, Any],
    private_key: ed25519.Ed25519PrivateKey,
    inline: bool = True,
) -> Dict[str, Any]:
    """
    Sign a JSON dictionary payload using RFC 8785 JCS and Ed25519-SHA512.
    
    If inline=True, injects the `_provenance_signature` envelope block directly.
    """
    # Create payload copy without previous signature envelope
    clean_payload = {k: v for k, v in payload.items() if k != "_provenance_signature"}
    canonical_bytes = canonicalize_json(clean_payload)
    
    sig_b64, pub_b64, fingerprint = sign_canonical_bytes(canonical_bytes, private_key)
    
    sig_block = {
        "algorithm": "Ed25519-SHA512",
        "signature": sig_b64,
        "public_key": pub_b64,
        "fingerprint": fingerprint,
        "signed_payload_sha512": hashlib.sha512(canonical_bytes).hexdigest(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    
    if inline:
        signed_result = dict(clean_payload)
        signed_result["_provenance_signature"] = sig_block
        return signed_result
    
    return sig_block


def create_detached_signature(
    payload: Dict[str, Any],
    private_key: ed25519.Ed25519PrivateKey,
    output_path: Union[str, Path],
) -> Path:
    """Generate a detached sidecar signature file (<job_id>.sig.json)."""
    sig_block = sign_report_payload(payload, private_key, inline=False)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sig_block, f, indent=2)
    return out_file


def verify_report_signature(
    report_path: Union[str, Path],
    public_key_fingerprint: Optional[str] = None,
    public_key_b64: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Verify the cryptographic Ed25519-SHA512 signature of a JSON report or payload file.
    
    Returns:
        (True, "VALID") on success, or (False, "[ERR_SIGNATURE_MISMATCH]: <reason>") on failure.
    """
    p = Path(report_path)
    if not p.exists():
        return False, f"[ERR_SIGNATURE_MISMATCH]: Report file not found at {p.as_posix()}"
    
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"[ERR_SIGNATURE_MISMATCH]: Failed to parse JSON report: {e}"
    
    # Locate signature block
    sig_block = None
    if isinstance(data, dict) and "_provenance_signature" in data:
        sig_block = data["_provenance_signature"]
        payload_to_verify = {k: v for k, v in data.items() if k != "_provenance_signature"}
    else:
        # Check sidecar .sig.json
        sidecar_path = p.with_suffix(".sig.json")
        if sidecar_path.exists():
            try:
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    sig_block = json.load(f)
                payload_to_verify = data
            except Exception as e:
                return False, f"[ERR_SIGNATURE_MISMATCH]: Failed to read sidecar signature: {e}"
        else:
            return False, "[ERR_SIGNATURE_MISMATCH]: No inline or sidecar signature envelope found"
            
    if not isinstance(sig_block, dict) or "signature" not in sig_block:
        return False, "[ERR_SIGNATURE_MISMATCH]: Malformed signature block"
        
    extracted_sig_b64 = sig_block.get("signature")
    extracted_pub_b64 = public_key_b64 or sig_block.get("public_key")
    extracted_fp = sig_block.get("fingerprint")
    
    if not extracted_pub_b64:
        return False, "[ERR_SIGNATURE_MISMATCH]: Public key missing in signature block"
        
    try:
        raw_pub = base64.urlsafe_b64decode(extracted_pub_b64.encode("ascii"))
        sig_bytes = base64.urlsafe_b64decode(extracted_sig_b64.encode("ascii"))
        pub_key = ed25519.Ed25519PublicKey.from_public_bytes(raw_pub)
    except Exception as e:
        return False, f"[ERR_SIGNATURE_MISMATCH]: Failed to decode public key or signature bytes: {e}"
        
    # Validate fingerprint if requested
    computed_fp = hashlib.sha256(raw_pub).hexdigest()
    if public_key_fingerprint and computed_fp != public_key_fingerprint:
        return False, f"[ERR_SIGNATURE_MISMATCH]: Public key fingerprint mismatch ({computed_fp} != {public_key_fingerprint})"
        
    if extracted_fp and computed_fp != extracted_fp:
        return False, f"[ERR_SIGNATURE_MISMATCH]: Envelope fingerprint mismatch ({computed_fp} != {extracted_fp})"
        
    # Re-canonicalize payload and verify
    canonical_bytes = canonicalize_json(payload_to_verify)
    sha512_digest = hashlib.sha512(canonical_bytes).digest()
    
    try:
        pub_key.verify(sig_bytes, sha512_digest)
        return True, "VALID"
    except InvalidSignature:
        return False, "[ERR_SIGNATURE_MISMATCH]: Cryptographic signature verification failed (payload altered)"
    except Exception as e:
        return False, f"[ERR_SIGNATURE_MISMATCH]: Verification error: {e}"


def save_private_key(private_key: ed25519.Ed25519PrivateKey, path: Union[str, Path]) -> None:
    """Save an Ed25519 private key to disk with strict 0600 file permissions."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    pem_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    
    with open(p, "wb") as f:
        f.write(pem_bytes)
        
    # Restrict permissions
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as err:
        logger.debug(f"Could not set POSIX 0600 on {p}: {err}")


def load_private_key(path: Union[str, Path]) -> ed25519.Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM file."""
    p = Path(path)
    with open(p, "rb") as f:
        pem_bytes = f.read()
    loaded = serialization.load_pem_private_key(pem_bytes, password=None)
    if not isinstance(loaded, ed25519.Ed25519PrivateKey):
        raise TypeError(f"Loaded key is not an Ed25519 private key: {type(loaded)}")
    return loaded
