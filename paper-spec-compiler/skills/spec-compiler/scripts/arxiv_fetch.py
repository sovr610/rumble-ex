#!/usr/bin/env python3
"""
arxiv_fetch.py — Download paper sources from arXiv.

Downloads PDF, LaTeX source tarball, and checks ar5iv HTML availability for a
given arXiv paper. Saves a manifest.json with full provenance information.

Usage:
    python arxiv_fetch.py --arxiv-id 2510.14783
    python arxiv_fetch.py --url https://arxiv.org/abs/2510.14783
    python arxiv_fetch.py --arxiv-id 2510.14783 --output-dir ./my_papers/
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print(
        "ERROR: 'requests' is required. Install it with: pip install requests",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("arxiv_fetch")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{id}"
AR5IV_HTML_URL = "https://ar5iv.labs.arxiv.org/html/{id}"

# arXiv ID pattern: optional "v<n>" version suffix is tolerated and stripped.
# Supports both old-style (quant-ph/0601001) and new-style (2510.14783).
ARXIV_ID_RE = re.compile(
    r"(?:(?:abs|pdf|e-print)/)?([a-zA-Z\-]+/\d{7}|\d{4}\.\d{4,5})(v\d+)?$",
    re.IGNORECASE,
)

CHUNK_SIZE = 1 << 15  # 32 KiB streaming chunks
REQUEST_TIMEOUT = 60  # seconds per request
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_session() -> requests.Session:
    """Return a requests Session with retry logic and a descriptive User-Agent."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "arxiv-fetch/1.0 (paper-spec-compiler; "
                "https://github.com/example/paper-spec-compiler)"
            )
        }
    )
    return session


def parse_arxiv_id(value: str) -> str:
    """
    Extract a canonical arXiv ID from a raw string that may be a full URL,
    a bare ID, or an ID with a version suffix.

    Examples accepted:
        2510.14783
        2510.14783v2
        https://arxiv.org/abs/2510.14783
        https://arxiv.org/pdf/2510.14783v1
        quant-ph/0601001

    Returns the ID without a version suffix (e.g. "2510.14783").
    Raises ValueError if the ID cannot be recognised.
    """
    # Strip whitespace and trailing slashes.
    value = value.strip().rstrip("/")

    # For full URLs, work on the path component only.
    parsed = urlparse(value)
    candidate = parsed.path if parsed.scheme else value

    match = ARXIV_ID_RE.search(candidate)
    if not match:
        raise ValueError(
            f"Cannot parse an arXiv ID from: {value!r}\n"
            "Expected formats: '2510.14783', 'quant-ph/0601001', "
            "or a full arxiv.org URL."
        )
    arxiv_id = match.group(1)
    version = match.group(2) or ""
    if version:
        log.info("Version suffix %r ignored; using base ID %r.", version, arxiv_id)
    return arxiv_id


def sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(n_bytes: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TiB"


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    description: str,
) -> dict:
    """
    Stream *url* to *dest*.  Returns a result dict describing the outcome.

    Result keys:
        success     bool
        url         str
        dest        str  (absolute path, set only on success)
        size_bytes  int  (set only on success)
        sha256      str  (set only on success)
        error       str  (set only on failure)
        timestamp   str  ISO-8601 UTC
    """
    result: dict = {
        "success": False,
        "url": url,
        "description": description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log.info("Fetching %s from %s", description, url)
    try:
        response = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            bytes_written = 0
            with dest.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        bytes_written += len(chunk)

            sha256 = sha256_of_file(dest)
            result.update(
                {
                    "success": True,
                    "dest": str(dest.resolve()),
                    "size_bytes": bytes_written,
                    "sha256": sha256,
                    "http_status": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                }
            )
            log.info(
                "  Saved %s (%s)  SHA-256: %s",
                dest.name,
                human_size(bytes_written),
                sha256[:16] + "…",
            )
        else:
            result["error"] = f"HTTP {response.status_code}"
            result["http_status"] = response.status_code
            log.warning("  %s returned HTTP %d.", description, response.status_code)

    except requests.exceptions.Timeout:
        result["error"] = "Request timed out"
        log.warning("  Timeout while fetching %s.", description)
    except requests.exceptions.ConnectionError as exc:
        result["error"] = f"Connection error: {exc}"
        log.warning("  Connection error for %s: %s", description, exc)
    except requests.exceptions.RequestException as exc:
        result["error"] = f"Request error: {exc}"
        log.warning("  Request failed for %s: %s", description, exc)
    except OSError as exc:
        result["error"] = f"I/O error: {exc}"
        log.error("  I/O error saving %s: %s", dest, exc)

    return result


def check_html_availability(
    session: requests.Session,
    url: str,
) -> dict:
    """
    Issue a HEAD request to check whether the ar5iv HTML page exists.
    Does NOT download the full page.

    Returns a result dict similar to download_file but without dest/sha256.
    """
    result: dict = {
        "success": False,
        "url": url,
        "description": "ar5iv HTML (availability check only)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log.info("Checking ar5iv HTML availability at %s", url)
    try:
        response = session.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        result["http_status"] = response.status_code
        if response.status_code == 200:
            result["success"] = True
            result["content_type"] = response.headers.get("Content-Type", "")
            log.info("  ar5iv HTML is available (HTTP 200).")
        else:
            result["error"] = f"HTTP {response.status_code}"
            log.info("  ar5iv HTML not available (HTTP %d).", response.status_code)
    except requests.exceptions.Timeout:
        result["error"] = "Request timed out"
        log.warning("  Timeout while checking ar5iv HTML.")
    except requests.exceptions.RequestException as exc:
        result["error"] = f"Request error: {exc}"
        log.warning("  Could not reach ar5iv: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def save_manifest(output_dir: Path, arxiv_id: str, results: dict) -> Path:
    """
    Write manifest.json to *output_dir* and return its path.

    The manifest contains:
        arxiv_id        canonical arXiv ID
        fetched_at      ISO-8601 UTC timestamp of this run
        sources         dict of per-source result records
    """
    manifest = {
        "arxiv_id": arxiv_id,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources": results,
    }
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    log.info("Manifest saved to %s", manifest_path.resolve())
    return manifest_path


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------


def fetch_paper(arxiv_id: str, output_dir: Path) -> dict:
    """
    Fetch all available sources for *arxiv_id* into *output_dir*.

    Returns a dict mapping source names to result records.
    """
    paper_dir = output_dir / arxiv_id.replace("/", "_")
    paper_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    results: dict = {}

    # --- PDF ---
    pdf_url = ARXIV_PDF_URL.format(id=arxiv_id)
    pdf_dest = paper_dir / f"{arxiv_id.replace('/', '_')}.pdf"
    results["pdf"] = download_file(session, pdf_url, pdf_dest, "PDF")

    # Polite delay between requests.
    time.sleep(1)

    # --- LaTeX source tarball ---
    src_url = ARXIV_EPRINT_URL.format(id=arxiv_id)
    # The e-print endpoint serves .tar.gz or .pdf; we use a generic extension
    # and let the content-type header inform us after download.
    src_dest = paper_dir / f"{arxiv_id.replace('/', '_')}_source.tar.gz"
    src_result = download_file(session, src_url, src_dest, "LaTeX source tarball")
    # Rename if the server returned a PDF instead of a tarball.
    if src_result["success"]:
        content_type = src_result.get("content_type", "")
        if "pdf" in content_type.lower():
            new_dest = src_dest.with_suffix("").with_suffix(".pdf")
            new_dest_path = new_dest
            src_dest.rename(new_dest_path)
            src_result["dest"] = str(new_dest_path.resolve())
            src_result["note"] = (
                "e-print endpoint returned a PDF (source not separately available)"
            )
            log.info(
                "  e-print returned PDF content; renamed to %s.", new_dest_path.name
            )
    results["source"] = src_result

    time.sleep(1)

    # --- ar5iv HTML (availability check only — the full page can be very large) ---
    html_url = AR5IV_HTML_URL.format(id=arxiv_id)
    results["html"] = check_html_availability(session, html_url)

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(arxiv_id: str, results: dict, manifest_path: Path) -> None:
    """Print a human-readable download summary to stdout."""
    divider = "-" * 60
    print()
    print(divider)
    print(f"  arXiv fetch summary for: {arxiv_id}")
    print(divider)

    labels = {
        "pdf": "PDF",
        "source": "LaTeX source tarball",
        "html": "ar5iv HTML",
    }

    for key, label in labels.items():
        rec = results.get(key, {})
        if rec.get("success"):
            if "dest" in rec:
                size_str = human_size(rec.get("size_bytes", 0))
                sha_short = rec.get("sha256", "")[:16]
                print(f"  [OK] {label}")
                print(f"       File : {rec['dest']}")
                print(f"       Size : {size_str}   SHA-256: {sha_short}…")
            else:
                # HTML availability check
                print(f"  [OK] {label} — available at {rec['url']}")
        else:
            err = rec.get("error", "unknown error")
            status = rec.get("http_status", "")
            status_str = f" (HTTP {status})" if status else ""
            print(f"  [--] {label} — not downloaded{status_str}: {err}")

    print()
    print(f"  Manifest : {manifest_path.resolve()}")
    print(divider)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arxiv_fetch",
        description=(
            "Download PDF, LaTeX source, and check HTML availability "
            "for an arXiv paper."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --arxiv-id 2510.14783
  %(prog)s --url https://arxiv.org/abs/2510.14783
  %(prog)s --arxiv-id 2510.14783 --output-dir ./papers/
  %(prog)s --arxiv-id quant-ph/0601001 --output-dir /tmp/arxiv/
        """,
    )

    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument(
        "--arxiv-id",
        metavar="ID",
        help="arXiv paper ID, e.g. '2510.14783' or 'quant-ph/0601001'.",
    )
    id_group.add_argument(
        "--url",
        metavar="URL",
        help="Full arXiv URL, e.g. 'https://arxiv.org/abs/2510.14783'.",
    )

    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=".paper_sources",
        help="Root directory for downloaded files (default: .paper_sources/).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve arXiv ID.
    raw = args.arxiv_id if args.arxiv_id else args.url
    try:
        arxiv_id = parse_arxiv_id(raw)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    log.info("arXiv ID resolved to: %s", arxiv_id)

    output_dir = Path(args.output_dir)

    # Fetch all sources.
    results = fetch_paper(arxiv_id, output_dir)

    # Save manifest.
    paper_dir = output_dir / arxiv_id.replace("/", "_")
    manifest_path = save_manifest(paper_dir, arxiv_id, results)

    # Print summary.
    print_summary(arxiv_id, results, manifest_path)

    # Exit code: 0 if at least the PDF was downloaded, 1 otherwise.
    if results.get("pdf", {}).get("success"):
        return 0
    else:
        log.error("PDF download failed; check the manifest for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
