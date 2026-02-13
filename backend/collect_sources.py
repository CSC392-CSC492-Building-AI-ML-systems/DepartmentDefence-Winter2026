#!/usr/bin/env python3
"""
Collect policy corpus sources for RAG.

Default behavior is "policy-first":
- Crawl Buyer's Guide pages
- Crawl Buy Canadian policy pages
- Fetch TBS Directive on the Management of Procurement (id=32692), including XML

Output:
- Plain text files in data/
- Manifest JSON in data/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup


BUYERS_GUIDE_ROOT = "https://canadabuys.canada.ca/en/buyer-s-portal/buyer-s-guide"
BUY_CANADIAN_ROOT = "https://canadabuys.canada.ca/en/buy-canadian-policy"
BUY_CANADIAN_POLICY_PREFIX = (
    "https://canadabuys.canada.ca/en/how-procurement-works/"
    "policies-and-guidelines/policies-directives-and-regulations"
)
FORMS_ROOT = (
    "https://canadabuys.canada.ca/en/buyer-s-portal/"
    "forms-templates-and-guides/forms-and-templates"
)
TBS_DIRECTIVE_HTML = "https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32692"
TBS_DIRECTIVE_XML = "https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32692&section=xml"

DEFAULT_SEEDS = [
    BUYERS_GUIDE_ROOT,
    BUY_CANADIAN_ROOT,
    TBS_DIRECTIVE_XML,
]

TBS_HOSTS = {
    "www.tbs-sct.canada.ca",
    "tbs-sct.canada.ca",
    "www.tbs-sct.gc.ca",
    "tbs-sct.gc.ca",
}
CANONICAL_TBS_HOST = "www.tbs-sct.canada.ca"

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

NOISE_EXACT = {
    "Language selection",
    "Canada.ca Menu",
    "Mid Level Menu",
    "CanadaBuys Menu",
    "You are here",
    "Search",
    "Top of Page",
}

NOISE_PREFIXES = (
    "Skip to main content",
    "Skip to \"About this site\"",
    "Buyer's portal homepage",
    "Explore the Buyer's Portal and send us your comments",
    "Search the Buyer's Guide",
    "Search the Buyer's Portal",
)


@dataclass
class FetchResult:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    body: str


@dataclass
class Document:
    doc_id: str
    title: str
    source_url: str
    canonical_url: str
    doc_type: str
    authority_rank: int
    date_modified: Optional[str]
    fetched_at: str
    content_hash: str
    word_count: int
    text_path: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_ws(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def keep_tbs_query(query_items: Iterable[Tuple[str, str]]) -> str:
    keep_keys = ("id", "section", "p")
    buckets: Dict[str, List[str]] = {k: [] for k in keep_keys}
    for k, v in query_items:
        if k in buckets:
            buckets[k].append(v)
    kept: List[Tuple[str, str]] = []
    for k in keep_keys:
        for v in buckets[k]:
            kept.append((k, v))
    return urlencode(kept, doseq=True)


def canonicalize_url(url: str) -> str:
    p = urlsplit(url.strip())
    scheme = p.scheme or "https"
    host = p.netloc.lower()
    if host in TBS_HOSTS:
        host = CANONICAL_TBS_HOST
    path = re.sub(r"/{2,}", "/", p.path or "/")
    if path != "/":
        path = path.rstrip("/")

    # Normalize known duplicate suffixes from canadabuys pages.
    if host == "canadabuys.canada.ca" and path.endswith("-0"):
        path = path[:-2]

    if host in TBS_HOSTS and path == "/pol/doc-eng.aspx":
        query = keep_tbs_query(parse_qsl(p.query, keep_blank_values=True))
    else:
        query = ""

    return urlunsplit((scheme, host, path, query, ""))


def in_scope(url: str, include_forms: bool, include_tbs_html: bool = False) -> bool:
    p = urlsplit(url)
    host = p.netloc.lower()
    path = p.path.rstrip("/")
    if not path:
        path = "/"

    if host == "canadabuys.canada.ca":
        if path == "/en/buy-canadian-policy":
            return True
        if path.startswith("/en/buyer-s-portal/buyer-s-guide"):
            return True
        if path.startswith(
            "/en/how-procurement-works/policies-and-guidelines/"
            "policies-directives-and-regulations"
        ):
            return True
        if include_forms and path.startswith(
            "/en/buyer-s-portal/forms-templates-and-guides/forms-and-templates"
        ):
            return True
        return False

    if host in TBS_HOSTS and path == "/pol/doc-eng.aspx":
        query = dict(parse_qsl(p.query, keep_blank_values=True))
        if query.get("id") != "32692":
            return False
        if include_tbs_html:
            return True
        return query.get("section") == "xml"

    return False


def classify_source(url: str) -> Tuple[str, int]:
    p = urlsplit(url)
    host = p.netloc.lower()
    path = p.path.rstrip("/")
    if not path:
        path = "/"

    if host in TBS_HOSTS and path == "/pol/doc-eng.aspx":
        return ("tbs_directive", 1)
    if path == "/en/buy-canadian-policy" or path.startswith(
        "/en/how-procurement-works/policies-and-guidelines/"
        "policies-directives-and-regulations"
    ):
        return ("buy_canadian_policy", 2)
    if path.startswith("/en/buyer-s-portal/buyer-s-guide"):
        return ("buyers_guide", 3)
    if path.startswith("/en/buyer-s-portal/forms-templates-and-guides/forms-and-templates"):
        return ("forms_templates", 4)
    return ("other", 9)


def should_follow_links(url: str, include_forms: bool) -> bool:
    p = urlsplit(url)
    host = p.netloc.lower()
    path = p.path.rstrip("/")
    if host != "canadabuys.canada.ca":
        return False

    if path.startswith("/en/buyer-s-portal/buyer-s-guide"):
        return True
    if path == "/en/buy-canadian-policy":
        return True
    if path.startswith(
        "/en/how-procurement-works/policies-and-guidelines/"
        "policies-directives-and-regulations"
    ):
        return True
    if include_forms and path.startswith(
        "/en/buyer-s-portal/forms-templates-and-guides/forms-and-templates"
    ):
        return True
    return False


def extract_links_from_html(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[str] = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        out.append(urljoin(base_url, href))
    return out


def pick_canadabuys_content_container(soup: BeautifulSoup) -> BeautifulSoup:
    # Buyer guide pages often put real content in the right region.
    preferred = soup.select_one(".bs-region--right")
    if preferred and len(" ".join(preferred.stripped_strings)) > 80:
        return preferred

    # Policy pages often expose content in region-content.
    region = soup.select_one("main .region-content")
    if region and len(" ".join(region.stripped_strings)) > 80:
        return region

    main = soup.find("main")
    if main:
        return main
    return soup


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if line in NOISE_EXACT:
        return True
    for prefix in NOISE_PREFIXES:
        if line.startswith(prefix):
            return True
    return False


def dedupe_extracted_lines(lines: List[str]) -> List[str]:
    """
    De-duplicate extracted text lines while preserving order.
    Also removes synthetic "combined" lines that fully contain other substantive lines.
    """
    # Remove exact duplicates globally (not only adjacent).
    unique: List[str] = []
    seen: Set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)

    out: List[str] = []
    for line in unique:
        # Drop a line if it appears to be a container concatenation of other
        # meaningful lines that are already present.
        # Example:
        # "Heading Description sentence..." plus separate "Heading" and
        # "Description sentence..." lines.
        is_combined = False
        for other in unique:
            if other == line:
                continue
            if len(other) < 40:
                continue
            if other in line and len(line) > len(other) + 20:
                is_combined = True
                break
        if is_combined:
            continue
        out.append(line)
    return out


def extract_date_modified(soup: BeautifulSoup, full_text: str) -> Optional[str]:
    # Prefer metadata fields first; page footers can contain stale template dates.
    for name in ("dcterms.modified", "dc.date.modified"):
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            text = normalize_ws(meta["content"])
            if re.match(r"\d{4}-\d{2}-\d{2}", text):
                return text

    # Common Canada.ca visible date blocks.
    for selector in (
        "time[property='dateModified']",
        "#wb-dtmd time",
    ):
        node = soup.select_one(selector)
        if node:
            text = normalize_ws(node.get_text(" ", strip=True))
            if re.match(r"\d{4}-\d{2}-\d{2}", text):
                return text

    match = re.search(r"Date modified:\s*(\d{4}-\d{2}-\d{2})", full_text, flags=re.I)
    if match:
        return match.group(1)
    return None


def extract_html_document(url: str, html: str) -> Tuple[str, str, Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")
    host = urlsplit(url).netloc.lower()
    container = pick_canadabuys_content_container(soup) if host == "canadabuys.canada.ca" else (
        soup.find("main") or soup
    )

    # Prefer page h1 when available.
    title = ""
    h1 = container.find("h1")
    if h1:
        title = normalize_ws(h1.get_text(" ", strip=True))
    if not title:
        meta_title = soup.find("meta", attrs={"name": "dc.title"}) or soup.find(
            "meta", attrs={"name": "dcterms.title"}
        )
        if meta_title and meta_title.get("content"):
            title = normalize_ws(meta_title["content"])
    if not title and soup.title:
        title = normalize_ws(soup.title.get_text(" ", strip=True))
    if not title:
        title = "Untitled"

    target_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]
    lines: List[str] = []
    for tag in container.find_all(target_tags):
        # Skip container elements that wrap other target elements to reduce duplicates.
        if tag.find(target_tags):
            continue
        text = normalize_ws(tag.get_text(" ", strip=True))
        if not text:
            continue
        if is_noise_line(text):
            continue
        lines.append(text)

    deduped = dedupe_extracted_lines(lines)
    text = normalize_ws("\n".join(deduped))

    page_text = normalize_ws(soup.get_text(" ", strip=True))
    date_modified = extract_date_modified(soup, page_text)
    return (title, text, date_modified)


def extract_tbs_xml_document(xml_text: str) -> Tuple[str, str, Optional[str]]:
    root = ET.fromstring(xml_text)
    title = normalize_ws(root.attrib.get("title", "Directive on the Management of Procurement"))
    date_modified = None

    # Extract date modified from notetoreader text if present.
    flat = normalize_ws(" ".join(root.itertext()))
    match = re.search(r"Date modified:\s*(\d{4}-\d{2}-\d{2})", flat, flags=re.I)
    if match:
        date_modified = match.group(1)

    lines: List[str] = []

    def walk(node: ET.Element) -> None:
        tag = node.tag.lower()
        if tag in {"chapter", "appendix", "section"}:
            node_title = normalize_ws(node.attrib.get("title", ""))
            anchor = node.attrib.get("anchor", "").strip()
            if node_title:
                if anchor:
                    lines.append(f"{anchor} {node_title}")
                else:
                    lines.append(node_title)

        if tag == "clause":
            anchor = node.attrib.get("anchor", "").strip()
            clause_text = normalize_ws(" ".join(node.itertext()))
            if clause_text:
                if anchor:
                    lines.append(f"{anchor} {clause_text}")
                else:
                    lines.append(clause_text)
            return

        for child in list(node):
            walk(child)

    walk(root)

    # De-duplicate adjacent lines.
    deduped: List[str] = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    text = normalize_ws("\n".join(deduped))
    return (title, text, date_modified)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    return s


def fetch_with_retries(
    session: requests.Session,
    url: str,
    timeout: int,
    retries: int,
    verbose: bool,
) -> Optional[FetchResult]:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code in {429, 500, 502, 503, 504}:
                if verbose:
                    print(f"[warn] {resp.status_code} for {url} (attempt {attempt}/{retries})")
                time.sleep(min(2 * attempt, 8))
                continue
            resp.raise_for_status()
            content_type = (resp.headers.get("Content-Type") or "").lower()
            return FetchResult(
                requested_url=url,
                final_url=resp.url,
                status_code=resp.status_code,
                content_type=content_type,
                body=resp.text,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if verbose:
                print(f"[warn] fetch failed for {url} (attempt {attempt}/{retries}): {exc}")
            time.sleep(min(2 * attempt, 8))
    if verbose and last_error:
        print(f"[error] giving up on {url}: {last_error}")
    return None


def filename_for_doc(doc_type: str, canonical_url: str) -> str:
    p = urlsplit(canonical_url)
    path_slug = re.sub(r"[^a-zA-Z0-9]+", "-", p.path.strip("/") or "root").strip("-").lower()
    query_slug = re.sub(r"[^a-zA-Z0-9]+", "-", p.query).strip("-").lower()
    base = f"{doc_type}__{path_slug}" if path_slug else doc_type
    if query_slug:
        base = f"{base}__{query_slug}"
    base = base[:140].strip("-")
    short_hash = hashlib.sha1(canonical_url.encode("utf-8")).hexdigest()[:10]
    return f"{base}__{short_hash}.txt"


def build_file_payload(doc: Document, body: str) -> str:
    header = [
        f"TITLE: {doc.title}",
        f"SOURCE_URL: {doc.source_url}",
        f"CANONICAL_URL: {doc.canonical_url}",
        f"DOC_TYPE: {doc.doc_type}",
        f"AUTHORITY_RANK: {doc.authority_rank}",
    ]
    if doc.date_modified:
        header.append(f"DATE_MODIFIED: {doc.date_modified}")
    header.append("---")
    return normalize_ws("\n".join(header) + "\n" + body + "\n")


def load_previous_manifest(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        docs = data.get("documents", [])
        return {d.get("canonical_url", ""): d for d in docs if d.get("canonical_url")}
    except Exception:  # noqa: BLE001
        return {}


def choose_seed_urls(include_forms: bool) -> List[str]:
    seeds = list(DEFAULT_SEEDS)
    if include_forms:
        seeds.append(FORMS_ROOT)
    return seeds


def choose_seed_urls_with_tbs(args: argparse.Namespace) -> List[str]:
    seeds = choose_seed_urls(args.include_forms)
    if args.include_tbs_html:
        seeds.append(TBS_DIRECTIVE_HTML)
    return seeds


def collect(args: argparse.Namespace) -> Dict:
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    previous = load_previous_manifest(manifest_path)
    session = make_session()

    queue: deque[str] = deque()
    raw_urls: Set[str] = set()
    seen_canonical: Set[str] = set()
    visited: Set[str] = set()
    docs: List[Document] = []
    errors: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []
    processed_final_canonical: Set[str] = set()

    for seed in choose_seed_urls_with_tbs(args):
        raw_urls.add(seed)
        c = canonicalize_url(seed)
        if in_scope(c, args.include_forms, args.include_tbs_html):
            queue.append(c)
            seen_canonical.add(c)

    while queue and len(visited) < args.max_pages:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        if args.verbose:
            print(f"[fetch] {current}")

        fetched = fetch_with_retries(
            session=session,
            url=current,
            timeout=args.timeout,
            retries=args.retries,
            verbose=args.verbose,
        )
        if not fetched:
            errors.append({"url": current, "error": "fetch_failed"})
            continue

        final_canonical = canonicalize_url(fetched.final_url)
        if not in_scope(final_canonical, args.include_forms, args.include_tbs_html):
            continue
        if final_canonical in processed_final_canonical:
            continue

        # Discover links for crawling.
        if "html" in fetched.content_type and should_follow_links(final_canonical, args.include_forms):
            links = extract_links_from_html(fetched.body, fetched.final_url)
            for link in links:
                raw_urls.add(link)
                c = canonicalize_url(link)
                if not in_scope(c, args.include_forms, args.include_tbs_html):
                    continue
                seen_canonical.add(c)
                if c not in visited:
                    queue.append(c)

        try:
            if "xml" in fetched.content_type or final_canonical.endswith("section=xml"):
                title, body_text, date_modified = extract_tbs_xml_document(fetched.body)
            else:
                title, body_text, date_modified = extract_html_document(final_canonical, fetched.body)
        except Exception as exc:  # noqa: BLE001
            errors.append({"url": current, "error": f"parse_failed: {exc}"})
            continue

        if not body_text:
            errors.append({"url": current, "error": "empty_text"})
            continue

        doc_type, authority_rank = classify_source(final_canonical)
        fetched_at = now_iso()
        content_hash = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
        word_count = len(body_text.split())
        if doc_type == "buyers_guide" and word_count < args.min_buyers_guide_words:
            skipped.append(
                {
                    "url": final_canonical,
                    "reason": f"buyers_guide_too_short<{args.min_buyers_guide_words}",
                    "word_count": str(word_count),
                }
            )
            processed_final_canonical.add(final_canonical)
            continue
        doc_id = hashlib.sha1(final_canonical.encode("utf-8")).hexdigest()[:12]
        filename = filename_for_doc(doc_type, final_canonical)
        text_path = out_dir / filename

        previous_doc = previous.get(final_canonical)
        changed = (
            args.force
            or previous_doc is None
            or previous_doc.get("content_hash") != content_hash
            or not text_path.exists()
        )

        record = Document(
            doc_id=doc_id,
            title=title,
            source_url=fetched.final_url,
            canonical_url=final_canonical,
            doc_type=doc_type,
            authority_rank=authority_rank,
            date_modified=date_modified,
            fetched_at=fetched_at,
            content_hash=content_hash,
            word_count=word_count,
            text_path=str(text_path.as_posix()),
        )

        if changed:
            payload = build_file_payload(record, body_text)
            text_path.write_text(payload, encoding="utf-8")
            if args.verbose:
                print(f"[write] {text_path} ({word_count} words)")
        elif args.verbose:
            print(f"[skip] unchanged {text_path}")

        docs.append(record)
        processed_final_canonical.add(final_canonical)
        time.sleep(args.sleep_seconds)

    # Ensure uniqueness by canonical URL in the final manifest.
    by_canonical: Dict[str, Document] = {}
    for d in docs:
        by_canonical[d.canonical_url] = d
    docs_sorted = sorted(by_canonical.values(), key=lambda d: (d.authority_rank, d.canonical_url))
    manifest = {
        "generated_at": now_iso(),
        "policy_first": True,
        "raw_url_count": len(raw_urls),
        "canonical_url_count": len(seen_canonical),
        "visited_count": len(visited),
        "document_count": len(docs_sorted),
        "errors": errors,
        "skipped": skipped,
        "documents": [d.__dict__ for d in docs_sorted],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect policy corpus sources for RAG.")
    p.add_argument(
        "--out-dir",
        default="data",
        help="Directory where .txt source files are written.",
    )
    p.add_argument(
        "--manifest",
        default="data/manifest.json",
        help="Path to manifest JSON.",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=300,
        help="Maximum number of canonical pages to visit.",
    )
    p.add_argument(
        "--include-forms",
        action="store_true",
        help="Also include forms/templates index pages (not enabled by default).",
    )
    p.add_argument(
        "--include-tbs-html",
        action="store_true",
        help="Also include TBS HTML page. By default, only TBS XML is collected.",
    )
    p.add_argument(
        "--min-buyers-guide-words",
        type=int,
        default=120,
        help="Skip Buyer’s Guide pages shorter than this threshold.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rewrite all files even if unchanged.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="HTTP timeout per request in seconds.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per request.",
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between fetches to reduce load on source servers.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-page progress.",
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    manifest = collect(args)
    print(
        "Collection complete: "
        f"{manifest['document_count']} documents, "
        f"{manifest['visited_count']} visited pages, "
        f"{manifest['canonical_url_count']} canonical URLs, "
        f"{manifest['raw_url_count']} raw URLs."
    )
    if manifest["errors"]:
        print(f"Warnings: {len(manifest['errors'])} pages had fetch/parse issues.")


if __name__ == "__main__":
    main()
