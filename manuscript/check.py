"""Executable policy checks for the NTCIR manuscript."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader
from paper.data import file_sha256

PROHIBITED_HARDWARE = ("L" + "40", "L" + "40S")
SOURCE_GLOBS = ("*.tex", "sections/*.tex", "*.bib")
REQUIRED_ASSETS = (
    "tables/table1_dataset.tex",
    "tables/table2_main.tex",
    "tables/table3_regimes.tex",
    "tables/table4_contrasts.tex",
    "figures/figure1_hierarchy.pdf",
)


def source_text(root: Path) -> str:
    files = sorted({path for pattern in SOURCE_GLOBS for path in root.glob(pattern) if path.is_file()})
    chunks = []
    for path in files:
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            continue
    return "\n".join(chunks)


def _active_source(text: str) -> str:
    """Remove TeX comments while preserving escaped percent signs."""
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def _metadata(root: Path) -> str:
    path = root / "metadata.tex"
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def source_errors(root: Path, final: bool = False) -> list[str]:
    text = source_text(root)
    active = _active_source(text)
    lowered = active.lower()
    errors: list[str] = []
    if any(marker.lower() in lowered for marker in PROHIBITED_HARDWARE):
        errors.append("manuscript contains a prohibited hardware reference")
    if re.search(r"\bno difference\b", lowered):
        errors.append("replace 'no difference' with 'no detectable difference'")
    if re.search(r"\bequivalence\b|\bequivalent\b", lowered):
        errors.append("an equivalence claim requires an equivalence design")
    inclusions = re.findall(r"\\(?:input|include)\s*\{([^}]+)\}", active)
    table4 = [target for target in inclusions if Path(target).name == "table4_contrasts.tex"]
    if not table4:
        errors.append("the full generated table4_contrasts.tex is not included")
    else:
        for target in table4:
            resolved = (root / target).resolve()
            if not resolved.is_file():
                errors.append(f"included generated asset is missing: {target}")
    metrics_present = ("path-constrained" in lowered or "hierarchical f1" in lowered
                       or re.search(r"\bc-wf1\b|\bhf\b", lowered) is not None)
    if metrics_present and "post hoc" not in lowered:
        errors.append("path-constrained wF1 and hierarchical F1 require a post hoc disclosure")
    metadata = _metadata(root)
    has_author = bool(re.search(r"\\author\{[^}]+\}", metadata))
    has_email = bool(re.search(r"\\email\{[^}]+@[^}]+\}", metadata))
    if final and not (has_author and has_email):
        errors.append("author metadata is required for final submission")
    return errors


def source_warnings(root: Path) -> list[str]:
    if re.search(r"\\author\{[^}]+\}", _metadata(root)):
        return []
    return ["author metadata is intentionally absent from this draft"]


def asset_errors(repo_root: Path) -> list[str]:
    errors = [f"required generated asset is missing: {path}" for path in REQUIRED_ASSETS
            if not (repo_root / path).is_file()]
    tables_manifest = repo_root / "tables" / "manifest.json"
    run_manifest = repo_root / "run_manifest.json"
    if not tables_manifest.is_file() or not run_manifest.is_file():
        return errors + ["generated assets lack committed provenance manifests"]
    try:
        table_data = json.loads(tables_manifest.read_text(encoding="utf-8"))
        run_data = json.loads(run_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return errors + ["generated asset provenance manifest is unreadable"]
    recorded_outputs = run_data.get("outputs", {}).get("tables", {})
    for relative in REQUIRED_ASSETS:
        if not relative.startswith("tables/"):
            continue
        name = Path(relative).name
        path = repo_root / relative
        recorded = recorded_outputs.get(name)
        if path.is_file() and recorded is None:
            errors.append(f"generated asset provenance is missing: {relative}")
        elif path.is_file() and recorded != file_sha256(path):
            errors.append(f"generated asset provenance checksum mismatch: {relative}")
    figure = repo_root / "figures" / "figure1_hierarchy.pdf"
    figure_recorded = run_data.get("outputs", {}).get("figures", {}).get(figure.name)
    if figure.is_file() and figure_recorded is not None and figure_recorded != file_sha256(figure):
        errors.append("generated asset provenance checksum mismatch: figures/figure1_hierarchy.pdf")
    for name, entry in table_data.get("tables", {}).items():
        for relative, recorded in entry.get("input_sha256", {}).items():
            path = repo_root / relative
            if not path.is_file() or file_sha256(path) != recorded:
                errors.append(f"generated asset provenance input mismatch: {relative}")
    return errors


def _read_pdf(pdf: Path):
    if not pdf.is_file():
        return None
    try:
        return PdfReader(str(pdf))
    except Exception:
        return None


def pdf_errors(pdf: Path, max_pages: int = 8) -> list[str]:
    if not pdf.is_file():
        return [f"compiled PDF is missing: {pdf}"]
    reader = _read_pdf(pdf)
    if reader is None:
        return [f"compiled PDF cannot be read: {pdf}"]
    pages = len(reader.pages)
    return [] if pages <= max_pages else [f"compiled PDF has {pages} pages; limit is {max_pages}"]


def font_errors(pdf: Path) -> list[str]:
    reader = _read_pdf(pdf)
    if reader is None:
        return [f"compiled PDF cannot be read for font checks: {pdf}"]
    missing: set[str] = set()
    for page in reader.pages:
        resources_ref = page.get("/Resources")
        if resources_ref is None:
            continue
        resources = resources_ref.get_object()
        fonts_ref = resources.get("/Font")
        if fonts_ref is None:
            continue
        for font_ref in fonts_ref.get_object().values():
            font = font_ref.get_object()
            if font.get("/Subtype") == "/Type3":
                continue
            descendants = font.get("/DescendantFonts") or []
            concrete = [item.get_object() for item in descendants] or [font]
            for item in concrete:
                descriptor_ref = item.get("/FontDescriptor")
                descriptor = descriptor_ref.get_object() if descriptor_ref else None
                embedded = descriptor is not None and any(
                    key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")
                )
                if not embedded:
                    missing.add(str(item.get("/BaseFont", font.get("/BaseFont", "unnamed font"))))
    return [f"font is not embedded: {name}" for name in sorted(missing)]


def log_errors(log: Path) -> list[str]:
    if not log.is_file():
        return [f"build log is missing: {log}"]
    text = log.read_text(encoding="utf-8", errors="replace").lower()
    errors = []
    for pattern in (r"there were undefined references", r"(?:latex|package natbib) warning: citation .* undefined"):
        if re.search(pattern, text):
            errors.append(f"build log matches unresolved-reference pattern: {pattern}")
    for width in re.findall(r"overfull \\hbox \(([\d.]+)pt too wide\)", text):
        if float(width) > 2.0:
            errors.append(f"build log has an overfull box {width}pt too wide")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    errors = source_errors(args.root, final=args.final)
    errors.extend(asset_errors(args.repo_root))
    if args.pdf:
        errors.extend(pdf_errors(args.pdf))
        errors.extend(font_errors(args.pdf))
    if args.log:
        errors.extend(log_errors(args.log))
    for error in errors:
        print(f"ERROR: {error}")
    for warning in source_warnings(args.root):
        print(f"WARNING: {warning}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
