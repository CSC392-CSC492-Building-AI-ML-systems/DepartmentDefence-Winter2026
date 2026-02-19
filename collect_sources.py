#!/usr/bin/env python3
"""
Collect policy corpus sources for RAG with structure-aware extraction.

Default behavior is "policy-first":
- Crawl Buyer's Guide pages
- Crawl Buy Canadian policy pages
- Fetch TBS Directive on the Management of Procurement HTML (id=32692)

Output:
- Markdown corpus files in data/corpus/
- Per-document metadata files in data/metadata/
- Manifest JSON in data/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup, Comment
from bs4.element import NavigableString, Tag


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
    TBS_DIRECTIVE_HTML,
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
    "Main Content",
    "Open Chat",
    "Report a problem on this page",
}

NOISE_PREFIXES = (
    "Skip to main content",
    "Skip to \"About this site\"",
    "Buyer's portal homepage",
    "Explore the Buyer's Portal and send us your comments",
    "Search the Buyer's Guide",
    "Search the Buyer's Portal",
    "Maintenance on CanadaBuys website",
    "SAP Ariba system maintenance",
    "Accessing Tenders in SAP Ariba",
)

GENERIC_DROP_SELECTORS = [
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "button",
    "input",
    "textarea",
    "select",
    "iframe",
    "footer",
    "aside",
    "header nav",
    ".breadcrumb",
    "#wb-bc",
    ".wb-inv",
    ".visually-hidden",
    ".sr-only",
    ".share-page",
    ".feedback",
    ".report-problem",
]

CANADABUYS_DROP_SELECTORS = [
    ".alertbox",
    ".sticky-side-nav",
    ".bs-region--left",
    ".nav-side",
]

TBS_DROP_SELECTORS = [
    "#def-preFooter",
    ".pagedetails",
    ".gc-prtts",
    ".pol-nav",
    ".pol-opt",
]

DROP_CLASS_KEYWORDS = (
    "alertbox",
    "sticky-side-nav",
    "bs-region--left",
    "nav-side",
    "share",
    "feedback",
    "search-api-page",
    "quickedit",
)

DROP_ID_KEYWORDS = (
    "def-prefooter",
    "wb-info",
    "wb-sec",
)

BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "blockquote", "pre"}


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


@dataclass
class ExtractedDocument:
    title: str
    markdown: str
    plain_text: str
    date_modified: Optional[str]
    breadcrumbs: List[Dict[str, Any]]
    blocks: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    extraction_method: str
    heading_count: int
    paragraph_count: int
    list_count: int
    table_count: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_inline(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_block(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned: List[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
            continue
        cleaned.append(line)
        prev_blank = False
    return "\n".join(cleaned).strip()


def normalize_ws(text: str) -> str:
    # Backward-compatible alias for code that expects generic normalization.
    return normalize_inline(text)


def tbs_query_to_canonical(query_items: Iterable[Tuple[str, str]]) -> str:
    query = dict(query_items)
    if query.get("id") != "32692":
        return ""
    section = (query.get("section") or "").lower()
    if section in {"", "html"}:
        return urlencode([("id", "32692")])
    if section == "xml":
        return urlencode([("id", "32692"), ("section", "xml")])

    kept: List[Tuple[str, str]] = [("id", "32692"), ("section", section)]
    p_value = query.get("p")
    if p_value:
        kept.append(("p", p_value))
    return urlencode(kept)


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
        query = tbs_query_to_canonical(parse_qsl(p.query, keep_blank_values=True))
    else:
        query = ""

    return urlunsplit((scheme, host, path, query, ""))


def in_scope(url: str, include_forms: bool, include_tbs_xml: bool = False) -> bool:
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
        section = (query.get("section") or "").lower()
        if section in {"", "html"}:
            return True
        return section == "xml" and include_tbs_xml

    return False


def classify_source(url: str) -> Tuple[str, int]:
    p = urlsplit(url)
    host = p.netloc.lower()
    path = p.path.rstrip("/")
    if not path:
        path = "/"

    if host in TBS_HOSTS and path == "/pol/doc-eng.aspx":
        section = dict(parse_qsl(p.query, keep_blank_values=True)).get("section", "").lower()
        if section == "xml":
            return ("tbs_directive_xml", 1)
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


def should_follow_links(url: str, include_forms: bool, include_tbs_xml: bool) -> bool:
    p = urlsplit(url)
    host = p.netloc.lower()
    path = p.path.rstrip("/")
    if host == "canadabuys.canada.ca":
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

    if host in TBS_HOSTS and path == "/pol/doc-eng.aspx":
        query = dict(parse_qsl(p.query, keep_blank_values=True))
        if query.get("id") != "32692":
            return False
        if (query.get("section") or "").lower() == "xml":
            return False
        return include_tbs_xml

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


def text_word_count(node: Tag) -> int:
    return len(normalize_inline(node.get_text(" ", strip=True)).split())


def select_content_nodes(soup: BeautifulSoup, url: str) -> Tuple[List[Tag], str]:
    p = urlsplit(url)
    host = p.netloc.lower()
    path = p.path.rstrip("/")

    nodes: List[Tag] = []
    method = "fallback_main"

    if host == "canadabuys.canada.ca":
        if path.startswith("/en/buyer-s-portal/buyer-s-guide"):
            top = soup.select_one(".bs-region--top")
            right = soup.select_one(".bs-region--right")
            if isinstance(top, Tag) and top.find(["h1", "h2"]):
                nodes.append(top)
            if isinstance(right, Tag) and text_word_count(right) >= 20:
                nodes.append(right)
                method = "buyers_guide_regions"
            else:
                region = soup.select_one("main .region-content")
                if isinstance(region, Tag):
                    nodes.append(region)
                    method = "buyers_guide_region_fallback"
        else:
            region = soup.select_one("main .region-content")
            if isinstance(region, Tag):
                nodes.append(region)
                method = "canadabuys_region_content"
    elif host in TBS_HOSTS and path == "/pol/doc-eng.aspx":
        ps_doc = soup.select_one("#ps-doc")
        if isinstance(ps_doc, Tag):
            nodes.append(ps_doc)
            method = "tbs_ps_doc"

    if not nodes:
        main = soup.find("main")
        if isinstance(main, Tag):
            nodes.append(main)
            method = "main"
        else:
            body = soup.find("body")
            if isinstance(body, Tag):
                nodes.append(body)
                method = "body"

    return (nodes, method)


def clone_and_sanitize(node: Tag, url: str) -> Optional[Tag]:
    cloned_soup = BeautifulSoup(str(node), "html.parser")
    root = cloned_soup.find()
    if not isinstance(root, Tag):
        return None

    p = urlsplit(url)
    selectors = list(GENERIC_DROP_SELECTORS)
    if p.netloc.lower() == "canadabuys.canada.ca":
        selectors.extend(CANADABUYS_DROP_SELECTORS)
    if p.netloc.lower() in TBS_HOSTS:
        selectors.extend(TBS_DROP_SELECTORS)

    for selector in selectors:
        for hit in list(root.select(selector)):
            hit.decompose()

    for comment in list(root.find_all(string=lambda text: isinstance(text, Comment))):
        comment.extract()

    for hit in list(root.find_all(True)):
        classes_blob = " ".join(hit.get("class") or []).lower()
        id_blob = (hit.get("id") or "").lower()
        if any(key in classes_blob for key in DROP_CLASS_KEYWORDS):
            hit.decompose()
            continue
        if any(key in id_blob for key in DROP_ID_KEYWORDS):
            hit.decompose()
            continue

    return root


def has_ancestor(tag: Tag, names: Set[str]) -> bool:
    parent = tag.parent
    while isinstance(parent, Tag):
        if (parent.name or "").lower() in names:
            return True
        parent = parent.parent
    return False


def iter_structural_tags(root: Tag) -> Iterable[Tag]:
    for tag in root.find_all(BLOCK_TAGS):
        name = (tag.name or "").lower()
        if name.startswith("h"):
            if has_ancestor(tag, {"table", "li", "nav", "aside"}):
                continue
        elif name == "p":
            if has_ancestor(tag, {"li", "table", "blockquote", "pre"}):
                continue
        elif name in {"ul", "ol"}:
            if has_ancestor(tag, {"li", "table", "blockquote", "pre"}):
                continue
        elif name == "table":
            if has_ancestor(tag, {"table"}):
                continue
        elif name in {"blockquote", "pre"}:
            if has_ancestor(tag, {"li", "table"}):
                continue
        yield tag


def inline_text_from_li(li: Tag) -> str:
    chunks: List[str] = []
    for child in li.contents:
        if isinstance(child, NavigableString):
            chunks.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = (child.name or "").lower()
        if name in {"ul", "ol", "table"}:
            continue
        chunks.append(child.get_text(" ", strip=True))
    return normalize_inline(" ".join(chunks))


def render_list_lines(tag: Tag, depth: int = 0, ordered: bool = False) -> Tuple[List[str], List[str]]:
    lines: List[str] = []
    plain_items: List[str] = []

    li_tags = tag.find_all("li", recursive=False)
    if not li_tags:
        li_tags = tag.find_all("li")

    idx = 1
    for li in li_tags:
        text = inline_text_from_li(li)
        if text:
            prefix = f"{idx}. " if ordered else "- "
            lines.append(("  " * depth) + prefix + text)
            plain_items.append(text)

        for nested in li.find_all(["ul", "ol"], recursive=False):
            nested_ordered = (nested.name or "").lower() == "ol"
            sub_lines, sub_plain = render_list_lines(nested, depth + 1, nested_ordered)
            lines.extend(sub_lines)
            plain_items.extend(sub_plain)

        idx += 1

    return (lines, plain_items)


def escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_table(tag: Tag) -> Optional[Dict[str, Any]]:
    rows_raw: List[List[str]] = []
    first_row_has_th = False
    for tr in tag.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        row = [normalize_inline(c.get_text(" ", strip=True)) for c in cells]
        if not any(row):
            continue
        if not rows_raw:
            first_row_has_th = any((c.name or "").lower() == "th" for c in cells)
        rows_raw.append(row)

    if not rows_raw:
        return None

    col_count = max(len(r) for r in rows_raw)
    rows = [r + [""] * (col_count - len(r)) for r in rows_raw]
    if first_row_has_th:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        headers = [f"Column {i + 1}" for i in range(col_count)]
        data_rows = rows

    md_lines = [
        "| " + " | ".join(escape_md_cell(c) for c in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in data_rows:
        md_lines.append("| " + " | ".join(escape_md_cell(c) for c in row) + " |")

    plain_rows = [" | ".join(r) for r in data_rows[:6]]
    plain = normalize_inline(" ; ".join([" | ".join(headers)] + plain_rows))
    return {
        "type": "table",
        "markdown": "\n".join(md_lines),
        "plain_text": plain,
        "table_meta": {
            "row_count": len(rows),
            "column_count": col_count,
            "headers": headers,
            "sample_rows": data_rows[:6],
        },
    }


def render_block_from_tag(tag: Tag) -> Optional[Dict[str, Any]]:
    name = (tag.name or "").lower()
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        text = normalize_inline(tag.get_text(" ", strip=True))
        if not text or is_noise_line(text):
            return None
        level = int(name[1])
        return {
            "type": "heading",
            "heading_level": level,
            "markdown": f"{'#' * level} {text}",
            "plain_text": text,
        }
    if name == "p":
        text = normalize_inline(tag.get_text(" ", strip=True))
        if not text or is_noise_line(text):
            return None
        return {"type": "paragraph", "markdown": text, "plain_text": text}
    if name in {"ul", "ol"}:
        lines, plain_items = render_list_lines(tag, depth=0, ordered=(name == "ol"))
        if not lines:
            return None
        markdown = "\n".join(lines)
        plain = normalize_inline(" ; ".join(plain_items))
        if not plain:
            return None
        return {"type": "list", "markdown": markdown, "plain_text": plain}
    if name == "table":
        return render_table(tag)
    if name == "blockquote":
        text = normalize_inline(tag.get_text(" ", strip=True))
        if not text:
            return None
        markdown = "\n".join(f"> {line}" for line in normalize_block(text).split("\n") if line)
        return {"type": "blockquote", "markdown": markdown, "plain_text": text}
    if name == "pre":
        raw = normalize_block(tag.get_text("\n", strip=True))
        if not raw:
            return None
        return {"type": "pre", "markdown": f"```\n{raw}\n```", "plain_text": normalize_inline(raw)}
    return None


def extract_title(nodes: List[Tag], soup: BeautifulSoup) -> str:
    for node in nodes:
        h1 = node.find("h1")
        if isinstance(h1, Tag):
            text = normalize_inline(h1.get_text(" ", strip=True))
            if text and not is_noise_line(text):
                return text

    h1 = soup.find("h1")
    if isinstance(h1, Tag):
        text = normalize_inline(h1.get_text(" ", strip=True))
        if text and not is_noise_line(text):
            return text

    meta_title = soup.find("meta", attrs={"name": "dc.title"}) or soup.find(
        "meta", attrs={"name": "dcterms.title"}
    )
    if isinstance(meta_title, Tag) and meta_title.get("content"):
        text = normalize_inline(meta_title["content"])
        if text:
            return text

    if soup.title:
        text = normalize_inline(soup.title.get_text(" ", strip=True))
        if text:
            return text
    return "Untitled"


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if line in NOISE_EXACT:
        return True
    for prefix in NOISE_PREFIXES:
        if line.startswith(prefix):
            return True
    return False


def extract_breadcrumbs(
    soup: BeautifulSoup,
    base_url: str,
    include_forms: bool,
    include_tbs_xml: bool,
) -> List[Dict[str, Any]]:
    candidates = [
        soup.select_one("#wb-bc"),
        soup.select_one("ol.breadcrumb"),
        soup.select_one(".breadcrumb"),
        soup.select_one("nav[aria-label*='breadcrumb' i]"),
    ]

    container: Optional[Tag] = None
    best_link_count = -1
    for cand in candidates:
        if not isinstance(cand, Tag):
            continue
        link_count = len(cand.find_all("a", href=True))
        if link_count > best_link_count:
            container = cand
            best_link_count = link_count
    if not isinstance(container, Tag):
        return []

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for a in container.find_all("a", href=True):
        label = normalize_inline(a.get_text(" ", strip=True))
        if not label:
            continue
        full_url = urljoin(base_url, a.get("href") or "")
        canonical = canonicalize_url(full_url)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(
            {
                "title": label,
                "url": full_url,
                "canonical_url": canonical,
                "in_scope": in_scope(canonical, include_forms, include_tbs_xml),
            }
        )
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


def assemble_document_structure(doc_id: str, title: str, raw_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    blocks_in: List[Dict[str, Any]] = []
    for raw in raw_blocks:
        block_type = str(raw.get("type") or "").strip().lower()
        if not block_type:
            continue
        markdown = normalize_block(str(raw.get("markdown") or ""))
        plain_text = normalize_inline(str(raw.get("plain_text") or ""))
        if not markdown or not plain_text:
            continue
        if block_type in {"heading", "paragraph"} and is_noise_line(plain_text):
            continue
        current: Dict[str, Any] = {
            "type": block_type,
            "markdown": markdown,
            "plain_text": plain_text,
        }
        if "heading_level" in raw:
            current["heading_level"] = int(raw["heading_level"])
        if "table_meta" in raw:
            current["table_meta"] = raw["table_meta"]
        if blocks_in and blocks_in[-1]["type"] == current["type"] and blocks_in[-1]["markdown"] == current["markdown"]:
            continue
        blocks_in.append(current)

    title_norm = normalize_inline(title)
    has_title_heading = any(
        b["type"] == "heading"
        and normalize_inline(re.sub(r"^#{1,6}\s*", "", b["markdown"]).strip()).lower() == title_norm.lower()
        for b in blocks_in
    )
    if title_norm and not has_title_heading:
        blocks_in.insert(
            0,
            {
                "type": "heading",
                "heading_level": 1,
                "markdown": f"# {title_norm}",
                "plain_text": title_norm,
            },
        )

    root_id = f"{doc_id}::root"
    root_path = [title_norm] if title_norm else []
    root_section: Dict[str, Any] = {
        "section_id": root_id,
        "parent_section_id": None,
        "level": 0,
        "heading": title_norm or "Document Root",
        "heading_path": root_path,
        "block_count": 0,
        "word_count": 0,
        "char_count": 0,
        "table_count": 0,
        "list_count": 0,
        "paragraph_count": 0,
        "first_block_index": None,
        "last_block_index": None,
        "preview": "",
        "prev_section_id": None,
        "next_section_id": None,
    }

    sections: List[Dict[str, Any]] = [root_section]
    section_by_id: Dict[str, Dict[str, Any]] = {root_id: root_section}
    stack: List[Dict[str, Any]] = [{"level": 0, "section_id": root_id, "heading_path": root_path}]

    out_blocks: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    heading_count = 0
    paragraph_count = 0
    list_count = 0
    table_count = 0

    for raw in blocks_in:
        block_type = raw["type"]
        if block_type == "heading":
            level = max(1, min(6, int(raw.get("heading_level") or 2)))
            while len(stack) > 1 and int(stack[-1]["level"]) >= level:
                stack.pop()
            parent = stack[-1]
            section_id = f"{doc_id}::s{len(sections):03d}"
            heading_text = raw["plain_text"]
            heading_path = list(parent["heading_path"]) + [heading_text] if parent["heading_path"] else [heading_text]
            section: Dict[str, Any] = {
                "section_id": section_id,
                "parent_section_id": parent["section_id"],
                "level": level,
                "heading": heading_text,
                "heading_path": heading_path,
                "block_count": 0,
                "word_count": 0,
                "char_count": 0,
                "table_count": 0,
                "list_count": 0,
                "paragraph_count": 0,
                "first_block_index": None,
                "last_block_index": None,
                "preview": "",
                "prev_section_id": None,
                "next_section_id": None,
            }
            sections.append(section)
            section_by_id[section_id] = section
            stack.append({"level": level, "section_id": section_id, "heading_path": heading_path})
            current_section_id = section_id
            heading_count += 1
        else:
            current_section_id = stack[-1]["section_id"]
            if block_type == "paragraph":
                paragraph_count += 1
            elif block_type == "list":
                list_count += 1
            elif block_type == "table":
                table_count += 1

        section = section_by_id[current_section_id]
        block_index = len(out_blocks)
        block: Dict[str, Any] = {
            "block_id": f"{doc_id}::b{block_index + 1:04d}",
            "section_id": current_section_id,
            "type": block_type,
            "heading_path": list(section["heading_path"]),
            "markdown": raw["markdown"],
            "plain_text": raw["plain_text"],
        }
        if block_type == "table":
            table_meta = raw.get("table_meta") or {}
            table_id = f"{doc_id}::t{len(tables) + 1:03d}"
            block["table_id"] = table_id
            tables.append(
                {
                    "table_id": table_id,
                    "section_id": current_section_id,
                    "row_count": int(table_meta.get("row_count") or 0),
                    "column_count": int(table_meta.get("column_count") or 0),
                    "headers": list(table_meta.get("headers") or []),
                    "sample_rows": list(table_meta.get("sample_rows") or []),
                }
            )
        out_blocks.append(block)

        words = len(raw["plain_text"].split())
        section["block_count"] += 1
        section["word_count"] += words
        section["char_count"] += len(raw["plain_text"])
        if block_type == "table":
            section["table_count"] += 1
        elif block_type == "list":
            section["list_count"] += 1
        elif block_type == "paragraph":
            section["paragraph_count"] += 1
        if section["first_block_index"] is None:
            section["first_block_index"] = block_index
        section["last_block_index"] = block_index
        if len(section["preview"]) < 240:
            section["preview"] = normalize_inline(f"{section['preview']} {raw['plain_text']}")[:240]

    ordered_non_root = [s["section_id"] for s in sections if s["section_id"] != root_id]
    for idx, section_id in enumerate(ordered_non_root):
        sec = section_by_id[section_id]
        sec["prev_section_id"] = ordered_non_root[idx - 1] if idx > 0 else None
        sec["next_section_id"] = ordered_non_root[idx + 1] if idx + 1 < len(ordered_non_root) else None

    markdown = "\n\n".join(b["markdown"] for b in out_blocks).strip()
    plain_text = normalize_inline(" ".join(b["plain_text"] for b in out_blocks))
    return {
        "markdown": markdown,
        "plain_text": plain_text,
        "blocks": out_blocks,
        "sections": sections,
        "tables": tables,
        "heading_count": heading_count,
        "paragraph_count": paragraph_count,
        "list_count": list_count,
        "table_count": table_count,
    }


def extract_html_document(
    url: str,
    html: str,
    doc_id: str,
    include_forms: bool,
    include_tbs_xml: bool,
) -> ExtractedDocument:
    soup = BeautifulSoup(html, "html.parser")
    nodes, method = select_content_nodes(soup, url)
    title = extract_title(nodes, soup)

    raw_blocks: List[Dict[str, Any]] = []
    for node in nodes:
        cleaned = clone_and_sanitize(node, url)
        if not isinstance(cleaned, Tag):
            continue
        for tag in iter_structural_tags(cleaned):
            block = render_block_from_tag(tag)
            if block:
                raw_blocks.append(block)

    assembled = assemble_document_structure(doc_id=doc_id, title=title, raw_blocks=raw_blocks)
    if not assembled["markdown"]:
        fallback_text = normalize_inline(soup.get_text(" ", strip=True))
        fallback_blocks: List[Dict[str, Any]] = []
        if fallback_text:
            fallback_blocks.append({"type": "paragraph", "markdown": fallback_text, "plain_text": fallback_text})
        assembled = assemble_document_structure(doc_id=doc_id, title=title, raw_blocks=fallback_blocks)

    page_text = normalize_inline(soup.get_text(" ", strip=True))
    date_modified = extract_date_modified(soup, page_text)
    breadcrumbs = extract_breadcrumbs(soup, url, include_forms, include_tbs_xml)
    return ExtractedDocument(
        title=title,
        markdown=assembled["markdown"],
        plain_text=assembled["plain_text"],
        date_modified=date_modified,
        breadcrumbs=breadcrumbs,
        blocks=assembled["blocks"],
        sections=assembled["sections"],
        tables=assembled["tables"],
        extraction_method=method,
        heading_count=assembled["heading_count"],
        paragraph_count=assembled["paragraph_count"],
        list_count=assembled["list_count"],
        table_count=assembled["table_count"],
    )


def extract_tbs_xml_document(xml_text: str, doc_id: str) -> ExtractedDocument:
    root = ET.fromstring(xml_text)
    title = normalize_inline(root.attrib.get("title", "Directive on the Management of Procurement"))
    date_modified = None

    flat = normalize_inline(" ".join(root.itertext()))
    match = re.search(r"Date modified:\s*(\d{4}-\d{2}-\d{2})", flat, flags=re.I)
    if match:
        date_modified = match.group(1)

    raw_blocks: List[Dict[str, Any]] = []

    def walk(node: ET.Element, depth: int) -> None:
        tag = node.tag.lower()
        if tag in {"chapter", "appendix", "section"}:
            node_title = normalize_inline(node.attrib.get("title", ""))
            anchor = normalize_inline(node.attrib.get("anchor", ""))
            if node_title:
                heading_text = normalize_inline(f"{anchor} {node_title}".strip())
                level = max(2, min(6, depth + 1))
                raw_blocks.append(
                    {
                        "type": "heading",
                        "heading_level": level,
                        "markdown": f"{'#' * level} {heading_text}",
                        "plain_text": heading_text,
                    }
                )
            for child in list(node):
                walk(child, depth + 1)
            return
        if tag == "clause":
            anchor = normalize_inline(node.attrib.get("anchor", ""))
            clause_text = normalize_inline(" ".join(node.itertext()))
            if clause_text:
                if anchor and not clause_text.startswith(anchor):
                    clause_text = f"{anchor} {clause_text}"
                raw_blocks.append(
                    {"type": "paragraph", "markdown": clause_text, "plain_text": clause_text}
                )
            return
        for child in list(node):
            walk(child, depth)

    walk(root, depth=1)
    assembled = assemble_document_structure(doc_id=doc_id, title=title, raw_blocks=raw_blocks)
    return ExtractedDocument(
        title=title,
        markdown=assembled["markdown"],
        plain_text=assembled["plain_text"],
        date_modified=date_modified,
        breadcrumbs=[],
        blocks=assembled["blocks"],
        sections=assembled["sections"],
        tables=assembled["tables"],
        extraction_method="tbs_xml_tree",
        heading_count=assembled["heading_count"],
        paragraph_count=assembled["paragraph_count"],
        list_count=assembled["list_count"],
        table_count=assembled["table_count"],
    )


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


def filename_for_doc(doc_type: str, canonical_url: str, ext: str = ".md") -> str:
    p = urlsplit(canonical_url)
    path_slug = re.sub(r"[^a-zA-Z0-9]+", "-", p.path.strip("/") or "root").strip("-").lower()
    query_slug = re.sub(r"[^a-zA-Z0-9]+", "-", p.query).strip("-").lower()
    base = f"{doc_type}__{path_slug}" if path_slug else doc_type
    if query_slug:
        base = f"{base}__{query_slug}"
    base = base[:130].strip("-")
    short_hash = hashlib.sha1(canonical_url.encode("utf-8")).hexdigest()[:10]
    return f"{base}__{short_hash}{ext}"


def choose_seed_urls(include_forms: bool, include_tbs_xml: bool) -> List[str]:
    seeds = list(DEFAULT_SEEDS)
    if include_forms:
        seeds.append(FORMS_ROOT)
    if include_tbs_xml:
        seeds.append(TBS_DIRECTIVE_XML)
    return seeds


def clear_directory(path: Path) -> None:
    if not path.exists():
        return

    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError(f"Refusing to clean filesystem root: {resolved}")

    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            shutil.rmtree(child)


def fallback_parent_by_path(target_url: str, known_urls: Set[str]) -> Optional[str]:
    tp = urlsplit(target_url)
    tpath = tp.path.rstrip("/")
    if not tpath:
        return None

    best: Optional[str] = None
    best_len = -1
    for candidate in known_urls:
        if candidate == target_url:
            continue
        cp = urlsplit(candidate)
        if cp.netloc.lower() != tp.netloc.lower():
            continue
        cpath = cp.path.rstrip("/")
        if not cpath or cpath == tpath:
            continue
        if tpath.startswith(cpath + "/") and len(cpath) > best_len:
            best = candidate
            best_len = len(cpath)
    return best


def lineage_for_url(url: str, parent_by_url: Dict[str, Optional[str]]) -> List[str]:
    chain: List[str] = []
    seen: Set[str] = set()
    current: Optional[str] = url
    while current and current not in seen:
        chain.append(current)
        seen.add(current)
        current = parent_by_url.get(current)
    chain.reverse()
    return chain


def attach_graph_metadata(docs: List[Dict[str, Any]]) -> None:
    known_urls = {d["canonical_url"] for d in docs}
    incoming: Dict[str, Set[str]] = defaultdict(set)

    for doc in docs:
        url = doc["canonical_url"]
        outgoing = [u for u in doc.get("outgoing_in_scope_links", []) if u in known_urls and u != url]
        unique_outgoing = sorted(set(outgoing))
        doc["outgoing_in_scope_links"] = unique_outgoing
        for linked in unique_outgoing:
            incoming[linked].add(url)

    parent_by_url: Dict[str, Optional[str]] = {}
    for doc in docs:
        url = doc["canonical_url"]
        parent_url: Optional[str] = None
        for crumb in reversed(doc.get("breadcrumbs", [])):
            candidate = crumb.get("canonical_url")
            if candidate and candidate != url and candidate in known_urls:
                parent_url = candidate
                break
        if not parent_url:
            parent_url = fallback_parent_by_path(url, known_urls)
        parent_by_url[url] = parent_url

    for url in list(parent_by_url.keys()):
        seen = {url}
        current = parent_by_url.get(url)
        while current:
            if current in seen:
                parent_by_url[url] = None
                break
            seen.add(current)
            current = parent_by_url.get(current)

    children_by_url: Dict[str, List[str]] = defaultdict(list)
    for url, parent_url in parent_by_url.items():
        if parent_url:
            children_by_url[parent_url].append(url)

    url_to_doc_id = {d["canonical_url"]: d["doc_id"] for d in docs}
    for doc in docs:
        url = doc["canonical_url"]
        parent_url = parent_by_url.get(url)
        child_urls = sorted(children_by_url.get(url, []))
        lineage_urls = lineage_for_url(url, parent_by_url)

        doc["parent_url"] = parent_url
        doc["parent_doc_id"] = url_to_doc_id.get(parent_url) if parent_url else None
        doc["child_urls"] = child_urls
        doc["child_doc_ids"] = [url_to_doc_id[c] for c in child_urls if c in url_to_doc_id]
        doc["lineage_urls"] = lineage_urls
        doc["lineage_doc_ids"] = [url_to_doc_id[u] for u in lineage_urls if u in url_to_doc_id]
        doc["depth"] = max(0, len(lineage_urls) - 1)
        doc["incoming_in_scope_links"] = sorted(incoming.get(url, set()))


def collect(args: argparse.Namespace) -> Dict:
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    corpus_dir = out_dir / "corpus"
    metadata_dir = out_dir / "metadata"

    if args.clean:
        clear_directory(out_dir)

    corpus_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    session = make_session()

    queue: deque[str] = deque()
    raw_urls: Set[str] = set()
    seen_canonical: Set[str] = set()
    visited: Set[str] = set()
    docs: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []
    processed_final_canonical: Set[str] = set()

    for seed in choose_seed_urls(args.include_forms, args.include_tbs_xml):
        raw_urls.add(seed)
        c = canonicalize_url(seed)
        if in_scope(c, args.include_forms, args.include_tbs_xml):
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
        if not in_scope(final_canonical, args.include_forms, args.include_tbs_xml):
            continue
        if final_canonical in processed_final_canonical:
            continue

        outgoing_in_scope: Set[str] = set()

        if "html" in fetched.content_type:
            links = extract_links_from_html(fetched.body, fetched.final_url)
            should_follow = should_follow_links(final_canonical, args.include_forms, args.include_tbs_xml)
            for link in links:
                raw_urls.add(link)
                c = canonicalize_url(link)
                if not in_scope(c, args.include_forms, args.include_tbs_xml):
                    continue
                outgoing_in_scope.add(c)
                seen_canonical.add(c)
                if should_follow and c not in visited:
                    queue.append(c)

        doc_id = hashlib.sha1(final_canonical.encode("utf-8")).hexdigest()[:12]
        try:
            if "xml" in fetched.content_type or final_canonical.endswith("section=xml"):
                extracted = extract_tbs_xml_document(fetched.body, doc_id=doc_id)
            else:
                extracted = extract_html_document(
                    final_canonical,
                    fetched.body,
                    doc_id=doc_id,
                    include_forms=args.include_forms,
                    include_tbs_xml=args.include_tbs_xml,
                )
        except Exception as exc:  # noqa: BLE001
            errors.append({"url": current, "error": f"parse_failed: {exc}"})
            continue

        if not extracted.markdown:
            errors.append({"url": current, "error": "empty_text"})
            continue

        doc_type, authority_rank = classify_source(final_canonical)
        word_count = len(extracted.plain_text.split())
        if doc_type == "buyers_guide" and args.min_buyers_guide_words > 0 and word_count < args.min_buyers_guide_words:
            skipped.append(
                {
                    "url": final_canonical,
                    "reason": f"buyers_guide_too_short<{args.min_buyers_guide_words}",
                    "word_count": str(word_count),
                }
            )
            processed_final_canonical.add(final_canonical)
            continue

        filename = filename_for_doc(doc_type, final_canonical, ext=".md")
        text_path = corpus_dir / filename
        metadata_path = metadata_dir / f"{doc_id}.json"

        docs.append(
            {
                "doc_id": doc_id,
                "title": extracted.title,
                "source_url": fetched.final_url,
                "canonical_url": final_canonical,
                "doc_type": doc_type,
                "authority_rank": authority_rank,
                "date_modified": extracted.date_modified,
                "fetched_at": now_iso(),
                "content_hash": hashlib.sha256(extracted.markdown.encode("utf-8")).hexdigest(),
                "word_count": word_count,
                "text_path": text_path.as_posix(),
                "metadata_path": metadata_path.as_posix(),
                "extraction_method": extracted.extraction_method,
                "breadcrumbs": extracted.breadcrumbs,
                "outgoing_in_scope_links": sorted(u for u in outgoing_in_scope if u != final_canonical),
                "structure": {
                    "block_count": len(extracted.blocks),
                    "section_count": len(extracted.sections),
                    "heading_count": extracted.heading_count,
                    "paragraph_count": extracted.paragraph_count,
                    "list_count": extracted.list_count,
                    "table_count": extracted.table_count,
                },
                "sections": extracted.sections,
                "tables": extracted.tables,
                "blocks": extracted.blocks,
                "markdown": extracted.markdown,
            }
        )

        processed_final_canonical.add(final_canonical)
        time.sleep(args.sleep_seconds)

    # Ensure uniqueness by canonical URL in the final manifest.
    by_canonical: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        by_canonical[d["canonical_url"]] = d
    docs_unique = list(by_canonical.values())
    attach_graph_metadata(docs_unique)
    docs_sorted = sorted(docs_unique, key=lambda d: (int(d["authority_rank"]), d["canonical_url"]))

    manifest_docs: List[Dict[str, Any]] = []
    for doc in docs_sorted:
        text_path = Path(doc["text_path"])
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(doc["markdown"].rstrip() + "\n", encoding="utf-8")

        metadata_path = Path(doc["metadata_path"])
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_payload = {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "source_url": doc["source_url"],
            "canonical_url": doc["canonical_url"],
            "doc_type": doc["doc_type"],
            "authority_rank": doc["authority_rank"],
            "date_modified": doc["date_modified"],
            "fetched_at": doc["fetched_at"],
            "content_hash": doc["content_hash"],
            "word_count": doc["word_count"],
            "text_path": doc["text_path"],
            "extraction_method": doc["extraction_method"],
            "breadcrumbs": doc["breadcrumbs"],
            "graph": {
                "parent_url": doc["parent_url"],
                "parent_doc_id": doc["parent_doc_id"],
                "child_urls": doc["child_urls"],
                "child_doc_ids": doc["child_doc_ids"],
                "lineage_urls": doc["lineage_urls"],
                "lineage_doc_ids": doc["lineage_doc_ids"],
                "depth": doc["depth"],
                "incoming_in_scope_links": doc["incoming_in_scope_links"],
                "outgoing_in_scope_links": doc["outgoing_in_scope_links"],
            },
            "structure": doc["structure"],
            "sections": doc["sections"],
            "tables": doc["tables"],
            "blocks": doc["blocks"],
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

        manifest_docs.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source_url": doc["source_url"],
                "canonical_url": doc["canonical_url"],
                "doc_type": doc["doc_type"],
                "authority_rank": doc["authority_rank"],
                "date_modified": doc["date_modified"],
                "fetched_at": doc["fetched_at"],
                "content_hash": doc["content_hash"],
                "word_count": doc["word_count"],
                "text_path": doc["text_path"],
                "metadata_path": doc["metadata_path"],
                "extraction_method": doc["extraction_method"],
                "structure": doc["structure"],
                "breadcrumbs": doc["breadcrumbs"],
                "parent_url": doc["parent_url"],
                "parent_doc_id": doc["parent_doc_id"],
                "child_urls": doc["child_urls"],
                "child_doc_ids": doc["child_doc_ids"],
                "lineage_urls": doc["lineage_urls"],
                "lineage_doc_ids": doc["lineage_doc_ids"],
                "depth": doc["depth"],
                "incoming_in_scope_links": doc["incoming_in_scope_links"],
                "outgoing_in_scope_links": doc["outgoing_in_scope_links"],
            }
        )

    manifest = {
        "generated_at": now_iso(),
        "collector_version": 2,
        "policy_first": True,
        "seed_urls": choose_seed_urls(args.include_forms, args.include_tbs_xml),
        "output_layout": {
            "out_dir": out_dir.as_posix(),
            "corpus_dir": corpus_dir.as_posix(),
            "metadata_dir": metadata_dir.as_posix(),
        },
        "raw_url_count": len(raw_urls),
        "canonical_url_count": len(seen_canonical),
        "visited_count": len(visited),
        "document_count": len(manifest_docs),
        "errors": errors,
        "skipped": skipped,
        "documents": manifest_docs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect policy corpus sources for RAG.")
    p.add_argument(
        "--out-dir",
        default="data",
        help="Directory where corpus and metadata are written.",
    )
    p.add_argument(
        "--manifest",
        default="data/manifest.json",
        help="Path to manifest JSON.",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=350,
        help="Maximum number of canonical pages to visit.",
    )
    p.add_argument(
        "--include-forms",
        action="store_true",
        help="Also include forms/templates index pages (not enabled by default).",
    )
    p.add_argument(
        "--include-tbs-xml",
        action="store_true",
        help="Also collect TBS XML (HTML is collected by default).",
    )
    p.add_argument(
        "--include-tbs-html",
        action="store_true",
        help="Deprecated compatibility flag. HTML is already collected by default.",
    )
    p.add_argument(
        "--min-buyers-guide-words",
        type=int,
        default=0,
        help="Skip Buyer’s Guide pages shorter than this threshold. Default 0 keeps all pages.",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete all current out-dir contents before collecting.",
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
