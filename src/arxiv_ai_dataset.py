#!/usr/bin/env python3
"""
Build a CSV dataset of arXiv "AI" paper abstracts straight from the official arXiv API.

Corpus   : papers whose category list (primary OR cross-list) contains at least one of the
           corpus categories (default: cs.AI, cs.LG, stat.ML + cs.CL, cs.CV, cs.RO, cs.NE;
           change them with --corpus-categories).
Label    : the primary category (<arxiv:primary_category> in the Atom feed).
Dates    : `published` = date of the FIRST version (v1), which is what the API's
           `submittedDate` filter also uses (GMT).

How it works
------------
1. The period [start, end] is cut into short windows (default 7 days, never crossing
   a year boundary).
2. For every window we query
       (cat:cs.AI OR cat:cs.LG OR cat:stat.ML) AND submittedDate:[start TO end]
   sorted by submittedDate (ascending) and keep at most --max-per-window papers.
3. Every raw HTTP response is cached (gzip) and every finished window is saved in its own
   JSON file, so the script is RESUMABLE: re-run the same command and it continues.
4. At the end all windows are merged, de-duplicated, cleaned and written to the CSV,
   followed by a summary (papers per year, top-20 primary categories, unmapped codes).

API etiquette (https://info.arxiv.org/help/api/tou.html)
--------------------------------------------------------
* at most one request every 3 seconds (enforced: --delay cannot go below 3 s for arxiv.org),
* a single connection at a time (one sequential requests.Session, no threads),
* identifiable User-Agent (--contact),
* retry with exponential backoff, honouring `Retry-After`,
* the API caps a single query at 30,000 results, hence the short windows.
* No PDFs are downloaded: only metadata and abstracts.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from arxiv_taxonomy import CATEGORY_INFO, describe

# --------------------------------------------------------------------------- constants
API_URL = "https://export.arxiv.org/api/query"
CORE_AI_CATEGORIES = ("cs.AI", "cs.LG", "stat.ML")      # the generic "catch-all" ML labels
GENERIC_AI = set(CORE_AI_CATEGORIES)                    # used for primary_is_generic_ai
DEFAULT_CORPUS_CATEGORIES = CORE_AI_CATEGORIES + ("cs.CL", "cs.CV", "cs.RO", "cs.NE")
API_MAX_START_PLUS_RESULTS = 30000                      # hard API ceiling per query
MIN_DELAY_ARXIV = 3.0                                   # seconds, from the Terms of Use

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}
CSV_COLUMNS = [
    "arxiv_id", "title", "abstract", "published", "year", "categories",
    "primary_category", "label_description", "label_area", "n_categories",
    "primary_is_generic_ai", "comment", "journal_ref",
]
ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(?:v(\d+))?$")
# Withdrawn papers: "This paper has been withdrawn by the author(s) ...", "Withdrawn due to ...".
# Only the first 200 characters are inspected, to avoid false positives in long, genuine abstracts.
WITHDRAWN_RE = re.compile(
    r"\b(?:has|have|had|is|was|been)\s+(?:been\s+)?withdrawn\b"
    r"|^\W*withdrawn\b"
    r"|\bwithdrawn\s+(?:by|due|because|from|pending|for)\b"
    r"|\bwithdraw(?:n)?\s+this\s+(?:paper|article|submission|manuscript|preprint)\b",
    re.IGNORECASE,
)

log = logging.getLogger("arxiv_ai")


class FatalError(RuntimeError):
    """Unrecoverable problem (e.g. too many failed retries). Re-running resumes the job."""


# --------------------------------------------------------------------------- logging
class TqdmHandler(logging.Handler):
    """Log through tqdm.write so log lines do not break the progress bar."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:  # pragma: no cover
            self.handleError(record)


def setup_logging(cache_dir: Path, verbose: bool):
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    console = TqdmHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    logfile = logging.FileHandler(cache_dir / "run.log", encoding="utf-8")
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(fmt)
    log.handlers[:] = [console, logfile]


# --------------------------------------------------------------------------- http client
class ArxivClient:
    """Sequential, rate-limited, caching client for the arXiv Atom API."""

    def __init__(self, api_url, contact, delay, timeout, retries, raw_dir: Path, offline=False,
                 backoff_base=10.0):
        self.api_url = api_url
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.raw_dir = raw_dir
        self.offline = offline
        self.backoff_base = backoff_base
        self._last_request_end = 0.0
        self.n_http = 0          # real network requests made in this run
        self.n_cached = 0        # responses served from the raw cache
        self.session = requests.Session()   # ONE connection at a time
        self.session.headers.update({
            "User-Agent": f"arxiv-ai-abstracts-dataset/1.0 (university text-mining project; {contact})",
            "Accept": "application/atom+xml",
        })

    # -- cache helpers
    def _key(self, params: dict) -> str:
        blob = json.dumps(params, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:20]

    def _cache_path(self, params: dict) -> Path:
        return self.raw_dir / f"{self._key(params)}.xml.gz"

    def _wait_for_slot(self):
        """Block until at least `delay` seconds passed since the end of the previous request."""
        wait = self._last_request_end + self.delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    # -- main entry point
    def query(self, params: dict):
        """Return (total_results, entries) for the query. Uses the cache when possible."""
        path = self._cache_path(params)
        if path.exists():
            try:
                text = gzip.decompress(path.read_bytes()).decode("utf-8")
                total, entries = parse_feed(text)
                self.n_cached += 1
                return total, entries
            except Exception as exc:  # corrupted cache file -> refetch
                log.warning("Corrupted cache file %s (%s); refetching", path.name, exc)
                path.unlink(missing_ok=True)
        if self.offline:
            raise FatalError(f"--offline: missing cached response for {params}")

        for attempt in range(self.retries + 1):
            self._wait_for_slot()
            retry_after = None
            try:
                resp = self.session.get(self.api_url, params=params, timeout=self.timeout)
                self.n_http += 1
                self._last_request_end = time.monotonic()
                if resp.status_code == 200:
                    total, entries = parse_feed(resp.text)       # may raise ValueError
                    expected = max(0, min(int(params["max_results"]), total - int(params["start"])))
                    if expected > 0 and len(entries) == 0:
                        raise ValueError("empty page although results exist (known transient arXiv glitch)")
                    if len(entries) < expected:
                        log.warning("Short page: got %d of %d expected entries (total=%d, start=%s)",
                                    len(entries), expected, total, params["start"])
                    self.raw_dir.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(gzip.compress(resp.text.encode("utf-8")))
                    os.replace(tmp, path)
                    return total, entries
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                # other 4xx: not transient (e.g. 400 for a malformed query) -> fail loudly
                raise FatalError(f"HTTP {resp.status_code} for {params}: {resp.text[:300]!r}")
            except FatalError:
                raise
            except (requests.RequestException, ValueError, ET.ParseError) as exc:
                self._last_request_end = time.monotonic()
                if attempt >= self.retries:
                    raise FatalError(f"giving up after {attempt + 1} attempts: {exc}") from exc
                backoff = min(self.backoff_base * 2 ** attempt, 300)
                if retry_after and retry_after.isdigit():
                    backoff = max(backoff, int(retry_after))
                log.warning("Request failed (%s). Retry %d/%d in %ds", exc, attempt + 1, self.retries, backoff)
                time.sleep(backoff)
        raise FatalError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------- parsing
def parse_feed(xml_text: str):
    """Parse an Atom page -> (totalResults, [entry dict, ...]). Raises ValueError on API errors."""
    root = ET.fromstring(xml_text)
    total_el = root.find("os:totalResults", NS)
    if total_el is None or not (total_el.text or "").strip().isdigit():
        raise ValueError("feed without opensearch:totalResults")
    total = int(total_el.text)
    entries = []
    for e in root.findall("atom:entry", NS):
        eid = (e.findtext("atom:id", "", NS) or "").strip()
        if "api/errors" in eid:      # arXiv reports errors as a one-entry feed
            raise ValueError(f"arXiv API error: {(e.findtext('atom:summary', '', NS) or '').strip()[:200]}")
        m = ID_RE.search(eid)
        if not m:
            log.warning("Unparseable id %r, skipped", eid)
            continue
        cats = [c.get("term") for c in e.findall("atom:category", NS) if c.get("term")]
        prim_el = e.find("arxiv:primary_category", NS)
        primary = prim_el.get("term") if prim_el is not None else (cats[0] if cats else "")
        entries.append({
            "arxiv_id": m.group(1),                       # version stripped
            "title": e.findtext("atom:title", "", NS) or "",
            "abstract": e.findtext("atom:summary", "", NS) or "",
            "published": (e.findtext("atom:published", "", NS) or "")[:10],   # v1 date
            "categories": cats,
            "primary_category": primary,
            "comment": e.findtext("arxiv:comment", "", NS) or "",       # free text written by the authors
            "journal_ref": e.findtext("arxiv:journal_ref", "", NS) or "",   # free text, often the venue
        })
    return total, entries


# --------------------------------------------------------------------------- windows
def make_windows(start: dt.date, end: dt.date, days: int):
    """Contiguous windows of `days` days; a window never crosses 31 Dec / 1 Jan."""
    s = start
    while s <= end:
        e = min(s + dt.timedelta(days=days - 1), end, dt.date(s.year, 12, 31))
        yield s, e
        s = e + dt.timedelta(days=1)


def window_query(s: dt.date, e: dt.date, categories) -> str:
    cats = " OR ".join(f"cat:{c}" for c in categories)
    # submittedDate = date of the first version, GMT, minute resolution (YYYYMMDDHHMM)
    return f"({cats}) AND submittedDate:[{s:%Y%m%d}0000 TO {e:%Y%m%d}2359]"


def window_file(win_dir: Path, s: dt.date, e: dt.date) -> Path:
    return win_dir / f"{s.isoformat()}_{e.isoformat()}.json"


def window_config(args) -> dict:
    """Settings that change what a window contains: a finished window is reused only if they match."""
    return {"categories": sorted(args.corpus_categories), "max_per_window": args.max_per_window,
            "sampling": args.sampling, "seed": args.seed if args.sampling == "random-offset" else None}


def download_window(client: ArxivClient, win_dir: Path, s, e, args):
    """Download (or skip, if already done) one window. Returns its summary dict."""
    path = window_file(win_dir, s, e)
    config = window_config(args)
    if path.exists():
        try:
            done = json.loads(path.read_text(encoding="utf-8"))
            if done.get("config") == config:
                return done
            log.warning("Window %s was downloaded with different settings, redoing it", path.name)
        except Exception:
            log.warning("Corrupted window file %s, redoing it", path.name)

    query = window_query(s, e, args.corpus_categories)
    base = {"search_query": query, "sortBy": "submittedDate", "sortOrder": "ascending"}

    # --- find out how many papers exist in the window and where to start reading
    if args.sampling == "random-offset":
        total, _ = client.query({**base, "start": 0, "max_results": 1})     # cheap probe
        n_target = min(total, args.max_per_window)
        rng = random.Random(f"{args.seed}|{s}|{e}")        # deterministic per window
        offset = rng.randint(0, total - n_target) if total > n_target else 0
    else:  # "first": the first page itself returns the total, no probe needed
        total, n_target, offset = None, None, 0

    records, start = [], offset
    while True:
        remaining = (n_target - len(records)) if n_target is not None else args.max_per_window
        if remaining <= 0:
            break
        size = min(args.page_size, remaining)
        if start + size > API_MAX_START_PLUS_RESULTS:
            log.warning("Window %s..%s exceeds the 30,000 API ceiling; use a shorter --window-days", s, e)
            break
        tot, entries = client.query({**base, "start": start, "max_results": size})
        if total is None:
            total = tot
            n_target = min(total, args.max_per_window)
        records.extend(entries)
        start += size
        if not entries or start >= total or len(records) >= n_target:
            break

    records = records[:n_target or 0]
    result = {"config": config, "start": s.isoformat(), "end": e.isoformat(), "total": total or 0,
              "offset": offset, "n_fetched": len(records), "sampling": args.sampling,
              "records": records}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)                                 # atomic checkpoint
    return result


# --------------------------------------------------------------------------- cleaning
def clean_text(text: str) -> str:
    """Only whitespace normalisation: newlines / repeated spaces -> one space. Nothing else."""
    return re.sub(r"\s+", " ", text or "").strip()


def is_withdrawn(abstract: str, comment: str) -> bool:
    head = abstract[:200]
    return bool(WITHDRAWN_RE.search(head)) or bool(re.match(r"\W*withdrawn\b", comment or "", re.I))


def build_rows(windows_data, args, start, end):
    """Merge the windows, de-duplicate, filter, and return (rows, stats, unmapped_counter)."""
    stats = Counter()
    seen, rows = set(), []
    unmapped = Counter()
    for w in windows_data:
        for r in w["records"]:
            stats["raw_records"] += 1
            if r["arxiv_id"] in seen:
                stats["duplicates"] += 1
                continue
            seen.add(r["arxiv_id"])
            cats = r["categories"]
            if not set(cats) & set(args.corpus_categories):   # safety net: corpus definition
                stats["not_in_corpus"] += 1
                continue
            published = r["published"]
            if not (start.isoformat() <= published <= end.isoformat()):
                stats["outside_date_range"] += 1
                continue
            abstract = clean_text(r["abstract"])
            if is_withdrawn(abstract, r["comment"]):
                stats["withdrawn"] += 1
                continue
            if len(abstract.split()) < args.min_words:
                stats["too_short"] += 1
                continue
            primary = r["primary_category"]
            desc, area, mapped = describe(primary)
            if not mapped:
                unmapped[primary] += 1
            rows.append({
                "arxiv_id": r["arxiv_id"],
                "title": clean_text(r["title"]),
                "abstract": abstract,
                "published": published,
                "year": int(published[:4]),
                "categories": " ".join(cats),
                "primary_category": primary,
                "label_description": desc,
                "label_area": area,
                "n_categories": len(cats),
                "primary_is_generic_ai": primary in GENERIC_AI,
                "comment": clean_text(r["comment"]),
                "journal_ref": clean_text(r.get("journal_ref", "")),
            })
    rows.sort(key=lambda x: (x["published"], x["arxiv_id"]))
    stats["final"] = len(rows)
    return rows, stats, unmapped


def write_csv(rows, path: Path, excel=False):
    """Standard CSV: comma-separated, UTF-8, fields with commas/quotes are quoted.
    excel=True: semicolon-separated + UTF-8 BOM, so Excel with Italian/European
    regional settings opens it correctly by double-click."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig" if excel else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, delimiter=";" if excel else ",")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- summary
def make_summary(rows, stats, unmapped, windows_data, args):
    lines = ["=" * 78, "SUMMARY", "=" * 78]
    lines.append(f"Cleaning: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    lines.append("")
    # per-year counts + sampling coverage (selected / existing papers in the windows)
    per_year = Counter(r["year"] for r in rows)
    existing, fetched, truncated = Counter(), Counter(), Counter()
    for w in windows_data:
        y = int(w["start"][:4])
        existing[y] += w["total"]
        fetched[y] += w["n_fetched"]
        truncated[y] += int(w["total"] > w["n_fetched"])
    lines.append("Papers per year (final CSV)   [papers existing in API / fetched / coverage / truncated windows]")
    for y in sorted(set(per_year) | set(existing)):
        cov = fetched[y] / existing[y] if existing[y] else float("nan")
        lines.append(f"  {y}: {per_year[y]:>6}      [{existing[y]:>7} / {fetched[y]:>6} / {cov:6.1%} / {truncated[y]:>2}]")
    lines.append(f"  TOTAL: {len(rows)}")
    lines.append("")
    top = Counter(r["primary_category"] for r in rows).most_common(20)
    lines.append("Top-20 primary categories")
    for code, n in top:
        desc, _area, _ = describe(code)
        lines.append(f"  {code:<18} {n:>6} ({n / max(1, len(rows)):5.1%})  {desc}")
    generic = sum(r["primary_is_generic_ai"] for r in rows)
    lines.append(f"\nprimary_is_generic_ai = True: {generic} ({generic / max(1, len(rows)):.1%})")
    lines.append("")
    if unmapped:
        lines.append("UNMAPPED primary category codes (fallback description used): "
                     + ", ".join(f"{c} ({n})" for c, n in unmapped.most_common()))
    else:
        lines.append("Unmapped primary category codes: none")
    # also report unmapped codes appearing only in the full category lists
    other = Counter(c for r in rows for c in r["categories"].split() if c not in CATEGORY_INFO)
    if other:
        lines.append("Codes not in the taxonomy table found in the full `categories` field: "
                     + ", ".join(f"{c} ({n})" for c, n in other.most_common()))
    lines.append("=" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------------------- cli / main
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download arXiv AI abstracts (cs.AI / cs.LG / stat.ML) via the official API into a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--start-year", type=int, default=2018, help="first year (window start = 1 Jan of that year)")
    p.add_argument("--start-date", type=dt.date.fromisoformat, default=None,
                   help="YYYY-MM-DD; overrides --start-year (handy for small tests)")
    p.add_argument("--end-date", type=dt.date.fromisoformat, default=dt.date(2026, 10, 15),
                   help="YYYY-MM-DD, inclusive. Keep it a few weeks in the past: new papers can still be reclassified")
    p.add_argument("--max-per-window", type=int, default=150,
                   help="max papers kept per window (150/week ~ 7,800/year)")
    p.add_argument("--window-days", type=int, default=7, help="window length in days")
    p.add_argument("--output", type=Path, default=Path("arxiv_ai_abstracts.csv"))
    p.add_argument("--cache-dir", type=Path, default=Path("arxiv_cache"))
    p.add_argument("--corpus-categories", default=",".join(DEFAULT_CORPUS_CATEGORIES),
                   help="comma-separated arXiv categories; a paper is in the corpus if it has at least one of them "
                        "(primary or cross-list). Changing it changes the query, hence the cache: use a new --cache-dir")
    p.add_argument("--checkpoint-every", type=int, default=100,
                   help="every N finished windows also write a partial CSV (<output>.partial.csv) built from the "
                        "windows downloaded so far, so you always have usable data on disk (0 = off)")
    p.add_argument("--min-words", type=int, default=30, help="drop abstracts shorter than this many words")
    p.add_argument("--sampling", choices=["random-offset", "first"], default="random-offset",
                   help="'first': first N papers of the window; 'random-offset': a random block of N consecutive "
                        "papers inside the window (less biased towards the first days, +1 request per window)")
    p.add_argument("--seed", type=int, default=42, help="seed of the random offsets")
    p.add_argument("--page-size", type=int, default=100, help="results per HTTP request (API max 2000)")
    p.add_argument("--contact", default=os.environ.get("ARXIV_CONTACT"),
                   help="your e-mail or project URL, put in the User-Agent (or set env ARXIV_CONTACT)")
    p.add_argument("--delay", type=float, default=3.2, help="seconds between requests (>= 3 for arxiv.org)")
    p.add_argument("--timeout", type=float, default=90, help="HTTP timeout in seconds")
    p.add_argument("--retries", type=int, default=6, help="retries per request (exponential backoff)")
    p.add_argument("--api-url", default=API_URL, help=argparse.SUPPRESS)   # used only by the local mock test
    p.add_argument("--backoff-base", type=float, default=10.0, help=argparse.SUPPRESS)  # tests only
    p.add_argument("--excel", action="store_true",
                   help="write ';'-separated CSV with UTF-8 BOM (for double-click opening in Excel with "
                        "European settings). Default: standard comma-separated UTF-8")
    p.add_argument("--offline", action="store_true", help="never use the network: only the raw cache")
    p.add_argument("--build-only", action="store_true", help="skip downloading; rebuild CSV from finished windows")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    a.corpus_categories = [c.strip() for c in a.corpus_categories.split(",") if c.strip()]
    if not a.corpus_categories:
        p.error("--corpus-categories is empty")
    if a.start_date is None:
        a.start_date = dt.date(a.start_year, 1, 1)
    if a.start_date > a.end_date:
        p.error("start is after --end-date")
    if a.window_days < 1 or a.max_per_window < 1 or not 1 <= a.page_size <= 2000:
        p.error("window-days/max-per-window must be >=1 and page-size in 1..2000")
    if urlparse(a.api_url).hostname and urlparse(a.api_url).hostname.endswith("arxiv.org"):
        if a.delay < MIN_DELAY_ARXIV:
            p.error(f"--delay must be >= {MIN_DELAY_ARXIV}s (arXiv Terms of Use)")
    if not (a.offline or a.build_only) and not a.contact:
        p.error("--contact is required (arXiv asks for an identifiable User-Agent), "
                "e.g. --contact you@university.edu")
    return a


def main(argv=None) -> int:
    args = parse_args(argv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    win_dir = args.cache_dir / "windows"
    win_dir.mkdir(exist_ok=True)
    setup_logging(args.cache_dir, args.verbose)

    windows = list(make_windows(args.start_date, args.end_date, args.window_days))
    if args.end_date > dt.date.today() - dt.timedelta(days=14):
        log.warning("--end-date is within the last 2 weeks: categories of very recent papers may still change")
    cfg = window_config(args)

    def _is_done(s, e):
        f = window_file(win_dir, s, e)
        try:
            return f.exists() and json.loads(f.read_text(encoding="utf-8")).get("config") == cfg
        except Exception:
            return False

    done = sum(_is_done(s, e) for s, e in windows)
    per_win_req = 1 + (args.sampling == "random-offset") * 1 + math.ceil(args.max_per_window / args.page_size) - 1
    todo = len(windows) - done
    log.info("Corpus categories: %s", ", ".join(args.corpus_categories))
    log.info("Period %s..%s | %d windows of <=%d days | %d already done, %d to do",
             args.start_date, args.end_date, len(windows), args.window_days, done, todo)
    log.info("Estimated worst case: ~%d requests, >= %.1f hours at %.1fs/request (plus server latency)",
             todo * per_win_req, todo * per_win_req * args.delay / 3600, args.delay)

    client = ArxivClient(args.api_url, args.contact or "offline", max(args.delay, 0.0), args.timeout,
                         args.retries, args.cache_dir / "raw", offline=args.offline,
                         backoff_base=args.backoff_base)

    windows_data = []
    new_windows = 0
    try:
        # `initial=done` makes the ETA reflect only the windows still to download
        with tqdm(total=len(windows), initial=0 if args.build_only else done, unit="win",
                  desc="windows", dynamic_ncols=True) as bar:
            for s, e in windows:
                if args.build_only:
                    path = window_file(win_dir, s, e)
                    if path.exists():
                        windows_data.append(json.loads(path.read_text(encoding="utf-8")))
                    bar.update(1)
                    continue
                was_done = _is_done(s, e)
                w = download_window(client, win_dir, s, e, args)
                windows_data.append(w)
                if was_done:
                    continue                      # already counted in bar's `initial`
                log.info("window %s..%s: %d existing, %d fetched (offset %d)",
                         s, e, w["total"], w["n_fetched"], w["offset"])
                bar.update(1)
                new_windows += 1
                if args.checkpoint_every and new_windows % args.checkpoint_every == 0:
                    part, _, _ = build_rows(windows_data, args, args.start_date, args.end_date)
                    write_csv(part, args.output.with_suffix(".partial.csv"), excel=args.excel)
                    log.info("checkpoint: partial CSV with %d rows saved", len(part))
                bar.set_postfix(papers=sum(x["n_fetched"] for x in windows_data), http=client.n_http)
    except KeyboardInterrupt:
        log.warning("Interrupted. Finished windows are saved: re-run the same command to resume.")
        return 130
    except FatalError as exc:
        log.error("%s\nFinished windows are saved: re-run the same command to resume.", exc)
        return 1

    rows, stats, unmapped = build_rows(windows_data, args, args.start_date, args.end_date)
    write_csv(rows, args.output, excel=args.excel)
    summary = make_summary(rows, stats, unmapped, windows_data, args)
    print("\n" + summary)
    args.output.with_suffix(".summary.txt").write_text(summary + "\n", encoding="utf-8")
    log.info("Wrote %s (%d rows). HTTP requests this run: %d, served from cache: %d",
             args.output, len(rows), client.n_http, client.n_cached)
    return 0


if __name__ == "__main__":
    sys.exit(main())
