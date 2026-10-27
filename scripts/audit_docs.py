"""Final audit of the three written deliverables.

Checks, in order:
  1. The files open cleanly with python-docx / python-pptx.
  2. No placeholder text (TODO, Lorem, XXX, Insert, ...).
  3. No broken images: every image relationship resolves inside the package
     and every image part starts with a valid PNG/JPEG/EMF/WMF signature.
  4. Every metric number matches Evidence/final/summary.md (numbers are
     collected from summary.md + README.md into an allowlist; anything else
     found in a document is flagged for review, and the "before" values may
     only appear in a sentence that says they are the old/v1 numbers).
  5. The PPT slide order matches the guide's slide-by-slide script.

Run:  .venv\\Scripts\\python.exe scripts\\audit_docs.py
Writes: Evidence/audit/audit_report.txt
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

REPORT = ROOT / "Docs" / "Project_Report_BlindAssist.docx"
PPTX = ROOT / "Presentation" / "BlindAssist_Presentation.pptx"
GUIDE = ROOT / "Docs" / "Presentation_Guide.docx"
SUMMARY = ROOT / "Evidence" / "final" / "summary.md"
README = ROOT / "README.md"
OUT = ROOT / "Evidence" / "audit" / "audit_report.txt"

# MSO_SHAPE_TYPE values used by python-pptx (13 = PICTURE, 6 = GROUP).
SHAPE_TYPE_PICTURE = 13
SHAPE_TYPE_GROUP = 6

PLACEHOLDER_RE = re.compile(
    r"\b(todo|tbd|tba|fixme|lorem ipsum|xxx|placeholder|insert here|"
    r"coming soon|your text here|sample text)\b",
    re.IGNORECASE,
)

# number tokens that may only be quoted as the OLD / before value
OLD_ONLY = {"75.6", "148.0", "13.2", "6.8", "71.6", "162.4", "38", "27"}
OLD_CONTEXT = re.compile(
    r"\b(before|v1|old|first run|earlier|historical|variance|version 1|"
    r"49\.6|115\.1|1)\b",
    re.IGNORECASE,
)

NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")

# headline facts that every deliverable must state the same way
FACTS_ALL = {
    "speed 320 px": ["49.6", "20.2"],
    "accuracy": ["0.6124", "0.6498", "0.5420"],
    "quality": ["234", "86"],
    "position": ["20"],
}
FACTS_DOC = {
    "report": {"speed 640 px": ["115.1", "8.7"], "conf": ["0.45"]},
    "guide": {"speed 640 px": ["115.1", "8.7"], "conf": ["0.45"]},
    "ppt": {},
}

# unit-anchored metric checks: the value written after a unit must come from
# summary.md (or README for the app's config constants)
METRIC_RULES = {
    "FPS": (
        r"(\d+(?:\.\d+)?)\s*(?:FPS|frames per second)",
        {"20.2", "8.7", "13.2", "6.8", "30", "20", "10", "8"},
    ),
    "ms": (
        r"(\d+(?:\.\d+)?)\s*ms\b",
        {
            "49.6",
            "115.1",
            "75.6",
            "148.0",
            "71.6",
            "162.4",
            "1.26",
            "46.39",
            "1.46",
            "2.57",
            "110.06",
            "1.65",
            "26.77",
            "36.27",
            "152.07",
            "1500",
            "1200",
            "450",
            "1000",
            "700",
            "800",
            "600",
            "60",
            "500",
            "100",
            "3000",
            "2000",
            "50",
            "20",
            "30",
            "40",
            "45",
            "55",
            "250",
            "350",
        },
    ),
    "mAP": (r"mAP(?:@0\.5(?::0\.95)?)?[^0-9]*(0\.\d{3,4})\b", {"0.6124", "0.4453"}),
    "precision": (r"[Pp]recision\s*[:=|]?\s*(0\.\d+)", {"0.6498"}),
    "recall": (r"[Rr]ecall\s*[:=|]?\s*(0\.\d+)", {"0.5420"}),
    "tests": (
        r"(?<![\d.])(\d+)\s*(?:tests|test cases)",
        {"234", "124", "44"},
    ),
    "coverage": (r"(?<![\d.])(\d+)\s*percent\s+coverage", {"86"}),
    "percent in brackets": (
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*%\s*\)",
        {"86", "92", "11", "83.3", "100"},
    ),
}


def docx_text(path: Path) -> tuple[str, int]:
    from docx import Document

    d = Document(str(path))
    chunks: list[str] = []
    for p in d.paragraphs:
        chunks.append(p.text)
    for t in d.tables:
        for row in t.rows:
            # one string per row so a cell keeps its header/row context
            chunks.append(" | ".join(c.text for c in row.cells))
    for s in d.sections:
        for p in s.header.paragraphs:
            chunks.append(p.text)
        for p in s.footer.paragraphs:
            chunks.append(p.text)
    return "\n".join(chunks), len(d.inline_shapes)


def pptx_text(path: Path) -> tuple[str, int, list[str]]:
    from pptx import Presentation

    pr = Presentation(str(path))
    chunks: list[str] = []
    titles: list[str] = []
    images = 0

    def walk(shape) -> None:
        nonlocal images
        if shape.shape_type == SHAPE_TYPE_PICTURE:
            images += 1
        if shape.has_table:
            for row in shape.table.rows:
                for cell in row.cells:
                    chunks.append(cell.text)
        if shape.has_text_frame and shape.text_frame.text.strip():
            chunks.append(shape.text_frame.text)
        if getattr(shape, "shape_type", None) == SHAPE_TYPE_GROUP:
            for sub in shape.shapes:
                walk(sub)

    for slide in pr.slides:
        first = None
        for sh in slide.shapes:
            walk(sh)
            if first is None and sh.has_text_frame and sh.text_frame.text.strip():
                first = sh.text_frame.text.strip().splitlines()[0]
        titles.append(first or "")
        if slide.has_notes_slide:
            chunks.append(slide.notes_slide.notes_text_frame.text)
    return "\n".join(chunks), images, titles


def image_check(path: Path) -> tuple[list[str], dict]:
    """Verify every image relationship resolves and has a valid signature."""
    problems: list[str] = []
    stats = {"images": 0, "bytes": 0}
    sig = {
        ".png": b"\x89PNG\r\n\x1a\n",
        ".jpg": b"\xff\xd8\xff",
        ".jpeg": b"\xff\xd8\xff",
        ".gif": b"GIF8",
        ".bmp": b"BM",
        ".emf": b"\x01\x00\x00\x00",
        ".wmf": b"\xd7\xcd\xc6\x9a",
        ".tiff": b"II*\x00",
    }
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for rel in sorted(n for n in names if n.endswith(".rels")):
            base = rel.rsplit("_rels/", 1)[0]
            try:
                # The XML comes from this project's own generated .docx/.pptx
                # files, not from an untrusted source.
                root = ET.fromstring(z.read(rel))  # noqa: S314
            except ET.ParseError as exc:  # pragma: no cover
                problems.append(f"{rel}: unreadable ({exc})")
                continue
            for node in root:
                if not node.get("Type", "").endswith("/image"):
                    continue
                target = node.get("Target", "")
                if "://" in target or node.get("TargetMode") == "External":
                    problems.append(f"{rel}: external image link {target}")
                    continue
                # resolve relative target
                parts = []
                for seg in (base + target).split("/"):
                    if seg == "..":
                        parts.pop() if parts else None
                    elif seg not in ("", "."):
                        parts.append(seg)
                member = "/".join(parts)
                if member not in names:
                    problems.append(f"{rel}: missing image part {member}")
                    continue
                stats["images"] += 1
                data = z.read(member)
                stats["bytes"] += len(data)
                ext = Path(member).suffix.lower()
                if len(data) == 0:
                    problems.append(f"{member}: zero-byte image")
                elif ext in sig and not data.startswith(sig[ext]):
                    problems.append(f"{member}: bad signature for {ext}")
        # orphaned media never referenced by any rels file
        media = {n for n in names if n.startswith(("ppt/media/", "word/media/"))}
        referenced = set()
        for rel in (n for n in names if n.endswith(".rels")):
            try:
                # This project's own package XML (see the note above).
                root = ET.fromstring(z.read(rel))  # noqa: S314
            except ET.ParseError:
                continue
            base = rel.rsplit("_rels/", 1)[0]
            for node in root:
                if not node.get("Type", "").endswith("/image"):
                    continue
                parts = []
                for seg in (base + node.get("Target", "")).split("/"):
                    if seg == "..":
                        parts.pop() if parts else None
                    elif seg not in ("", "."):
                        parts.append(seg)
                referenced.add("/".join(parts))
        orphans = media - referenced
        if orphans:
            problems.append(f"orphaned media (not shown anywhere): {sorted(orphans)}")
    return problems, stats


def numbers(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in NUMBER_RE.finditer(text):
        tok = m.group(0)
        out[tok] = out.get(tok, 0) + 1
    return out


def sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+|\n+", text)


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []
    log: list[str] = ["BlindAssist - final document audit", ""]

    # ---- allowlist from summary.md + README.md ---------------------------
    allowed = set(NUMBER_RE.findall(SUMMARY.read_text("utf-8")))
    allowed |= set(NUMBER_RE.findall(README.read_text("utf-8")))
    log.append(
        f"allowlist of known numbers (summary.md + README.md): "
        f"{len(allowed)} tokens"
    )

    # ---- 1. open the files ------------------------------------------------
    docs: dict[str, str] = {}
    try:
        docs["report"], n_report_shapes = docx_text(REPORT)
        log.append(
            f"report : opens OK, {len(docs['report'].split())} words, "
            f"{n_report_shapes} inline pictures"
        )
    except Exception as exc:
        problems.append(f"report does not open: {exc}")
        n_report_shapes = 0
    try:
        docs["ppt"], n_ppt_pics, titles = pptx_text(PPTX)
        log.append(
            f"ppt    : opens OK, {len(docs['ppt'].split())} words, "
            f"{n_ppt_pics} pictures, {len(titles)} slides"
        )
    except Exception as exc:
        problems.append(f"ppt does not open: {exc}")
        n_ppt_pics, titles = 0, []
    try:
        docs["guide"], n_guide_shapes = docx_text(GUIDE)
        log.append(
            f"guide  : opens OK, {len(docs['guide'].split())} words, "
            f"{n_guide_shapes} inline pictures"
        )
    except Exception as exc:
        problems.append(f"guide does not open: {exc}")
        n_guide_shapes = 0

    # ---- 2. placeholders --------------------------------------------------
    log.append("")
    for name, text in docs.items():
        hits = sorted({m.group(0) for m in PLACEHOLDER_RE.finditer(text)})
        if hits:
            problems.append(f"{name}: placeholder text {hits}")
        log.append(f"placeholders in {name}: {hits or 'none'}")

    # ---- 3. images --------------------------------------------------------
    log.append("")
    for label, path in (("report", REPORT), ("ppt", PPTX), ("guide", GUIDE)):
        if not path.exists():
            problems.append(f"{label}: file missing at {path}")
            continue
        probs, stats = image_check(path)
        problems += [f"{label}: {p}" for p in probs]
        log.append(
            f"images in {label}: {stats['images']} image parts, "
            f"{stats['bytes'] / 1024:.0f} KB, "
            f"{len(probs)} problem(s)"
        )
        for p in probs:
            log.append(f"    - {p}")

    # ---- 4. numbers vs summary.md ----------------------------------------
    log.append("")
    for name, text in docs.items():
        toks = numbers(text)
        unknown = sorted(t for t in toks if t not in allowed)
        log.append(
            f"{name}: {len(toks)} distinct numbers, "
            f"{len(unknown)} not in summary.md/README: {unknown}"
        )
        if unknown:
            notes.append(f"{name}: review these tokens: {unknown}")
        # old/before values must be labelled as old
        for s in sentences(text):
            for tok in OLD_ONLY:
                if tok in toks and re.search(
                    rf"(?<![\d.]){re.escape(tok)}(?![\d.])", s
                ):
                    if not OLD_CONTEXT.search(s):
                        notes.append(
                            f"{name}: '{tok}' in a sentence with no "
                            f"'before/v1' context: "
                            f"{s.strip()[:120]!r}"
                        )

    # every headline fact must be present in every document
    log.append("")
    for name, text in docs.items():
        want = dict(FACTS_ALL)
        want.update(FACTS_DOC.get(name, {}))
        missing = [k for k, vals in want.items() if not all(v in text for v in vals)]
        if missing:
            problems.append(f"{name}: missing headline facts {missing}")
        log.append(
            f"{name}: headline facts present: "
            f"{sorted(want) if not missing else 'MISSING ' + str(missing)}"
        )

    # unit-anchored metric values must be values that summary.md allows
    log.append("")
    for name, text in docs.items():
        for label, (pattern, allowed_vals) in METRIC_RULES.items():
            found = sorted(set(re.findall(pattern, text)))
            bad = [v for v in found if v not in allowed_vals]
            log.append(
                f"{name}: {label:>9} values in text: {found or '-'}"
                + (f"  ** NOT ALLOWED: {bad}" if bad else "")
            )
            if bad:
                problems.append(
                    f"{name}: {label} value(s) {bad} not in " f"summary.md/README"
                )

    # honesty checks - the docs must not claim a speed-up
    log.append("")
    for name, text in docs.items():
        low = text.lower()
        has_variance = (
            "machine variance" in low
            or "measurement variance" in low
            or "did not really change" in low
            or "no code" in low
            or "unchanged" in low
        )
        log.append(
            f"{name}: states the speed/accuracy honesty caveat: " f"{has_variance}"
        )

    # ---- 5. PPT order vs guide script ------------------------------------
    log.append("")
    import guide_content as gc

    scripts = gc.SCRIPTS
    if len(titles) != len(scripts):
        problems.append(
            f"slide count mismatch: ppt {len(titles)} vs " f"guide {len(scripts)}"
        )
    keywords = {
        1: ["blindassist", "machine learning lab"],
        2: ["problem"],
        3: ["matters"],
        4: ["objectives"],
        5: ["existing solutions"],
        6: ["solution in one picture"],
        7: ["architecture"],
        8: ["yolo"],
        9: ["coco"],
        10: ["position and distance"],
        11: ["smart announcements"],
        12: ["improvements"],
        13: ["speech"],
        14: ["screenshots"],
        15: ["results"],
        16: ["limitations"],
        17: ["conclusion"],
    }
    order_ok = True
    for i, (title, sc) in enumerate(zip(titles, scripts, strict=False), start=1):
        key = keywords[sc["n"]]
        ok = any(k in title.lower() for k in key) and sc["n"] == i
        log.append(
            f"  slide {i:>2} | deck: {title[:44]:<44} | guide "
            f"script {sc['n']:>2}: {sc['title'][:40]:<40} | "
            f"{'MATCH' if ok else 'MISMATCH'}"
        )
        if not ok:
            order_ok = False
            problems.append(
                f"slide {i}: deck title {title!r} does not match "
                f"guide script {sc['n']}"
            )
    if order_ok:
        log.append("  -> slide order matches the guide's script 1..17")

    # ---- result -----------------------------------------------------------
    log.append("")
    log.append(f"problems : {len(problems)}")
    log.append(f"notes    : {len(notes)} (tokens to eyeball, not failures)")
    log.append("")
    for p in problems:
        log.append(f"- PROBLEM: {p}")
    for n in notes:
        log.append(f"- note   : {n}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(log), "utf-8")
    print("\n".join(log))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
