"""Executable policy checks for the NTCIR manuscript."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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
    "tables/table5_metrics.tex",
    "figures/figure1_hierarchy.pdf",
)
CANONICAL_FROZEN_ASSETS = REQUIRED_ASSETS + ("tables/manifest.json",)
LAYOUT_PLACEHOLDER_AUTHORS = frozenset(
    f"Student Author {number}" for number in range(1, 5)
)
LAYOUT_PLACEHOLDER_TODO_MARKERS = ("TODO(author-metadata)",)


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


def _has_layout_author_placeholders(metadata: str) -> bool:
    authors = re.findall(r"\\author\{([^}]*)\}", _active_source(metadata))
    return (
        any(author.strip() in LAYOUT_PLACEHOLDER_AUTHORS for author in authors)
        or any(marker in metadata for marker in LAYOUT_PLACEHOLDER_TODO_MARKERS)
    )


def source_errors(root: Path, final: bool = False, repo_root: Path | None = None) -> list[str]:
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
        canonical_table4 = ((repo_root or root) / "tables" / "table4_contrasts.tex").resolve()
        for target in table4:
            resolved = (root / target).resolve()
            if resolved != canonical_table4:
                errors.append(
                    f"Table 4 inclusion must resolve to canonical generated asset: {target}"
                )
            elif not resolved.is_file():
                errors.append(f"included generated asset is missing: {target}")
    metrics_present = ("path-constrained" in lowered or "hierarchical f1" in lowered
                       or re.search(r"\bc-wf1\b|\bhf\b", lowered) is not None)
    if metrics_present and "post hoc" not in lowered:
        errors.append("path-constrained wF1 and hierarchical F1 require a post hoc disclosure")
    metadata = _metadata(root)
    active_metadata = _active_source(metadata)
    has_author = bool(re.search(r"\\author\{[^}]+\}", active_metadata))
    has_email = bool(re.search(r"\\email\{[^}]+@[^}]+\}", active_metadata))
    if final and _has_layout_author_placeholders(metadata):
        errors.append("layout-only author placeholders must be replaced before final submission")
    elif final and not (has_author and has_email):
        errors.append("author metadata is required for final submission")
    return errors


def source_warnings(root: Path) -> list[str]:
    metadata = _metadata(root)
    if _has_layout_author_placeholders(metadata):
        return ["layout-only author placeholders remain in this draft"]
    if re.search(r"\\author\{[^}]+\}", _active_source(metadata)):
        return []
    return ["author metadata is intentionally absent from this draft"]


def _manifest_input_path(repo_root: Path, relative: str) -> tuple[Path | None, str | None]:
    candidate = Path(relative)
    if candidate.is_absolute():
        return None, f"generated asset provenance input path must be repository-relative: {relative}"
    if ".." in candidate.parts:
        return None, (
            "generated asset provenance input path must not traverse parent directories: "
            f"{relative}"
        )
    resolved_root = repo_root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None, f"generated asset provenance input path resolves outside repository: {relative}"
    return resolved, None


def asset_errors(repo_root: Path) -> list[str]:
    errors = [f"required generated asset is missing: {path}" for path in REQUIRED_ASSETS
              if not (repo_root / path).is_file()]
    for relative in CANONICAL_FROZEN_ASSETS:
        if not (repo_root / relative).is_file():
            continue
        try:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative],
                cwd=repo_root, capture_output=True, check=False,
            ).returncode == 0
            clean = subprocess.run(
                ["git", "diff", "--quiet", "main", "--", relative],
                cwd=repo_root, capture_output=True, check=False,
            ).returncode == 0
        except OSError:
            tracked, clean = False, False
        if not tracked:
            errors.append(f"canonical generated asset is not tracked: {relative}")
        elif not clean:
            errors.append(f"canonical generated asset is not clean relative to main: {relative}")
    tables_manifest = repo_root / "tables" / "manifest.json"
    run_manifest = repo_root / "run_manifest.json"
    if not tables_manifest.is_file() or not run_manifest.is_file():
        return errors + ["generated assets lack committed provenance manifests"]
    try:
        table_data = json.loads(tables_manifest.read_text(encoding="utf-8"))
        run_data = json.loads(run_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return errors + ["generated asset provenance manifest is unreadable"]
    if not isinstance(table_data, dict):
        errors.append("table manifest top level must be an object")
        table_data = {}
    if not isinstance(run_data, dict):
        errors.append("run manifest top level must be an object")
        run_data = {}
    required_tables = [Path(path).name for path in REQUIRED_ASSETS if path.startswith("tables/")]
    manifest_tables = table_data.get("tables", {})
    if not isinstance(manifest_tables, dict):
        errors.append("tables section must be an object")
        manifest_tables = {}
    for name in required_tables:
        entry = manifest_tables.get(name)
        if not isinstance(entry, dict):
            errors.append(f"required table manifest entry is missing: {name}")
        elif not entry.get("source_script") or not entry.get("input_files"):
            errors.append(f"required table manifest entry is incomplete: {name}")
    for name, entry in manifest_tables.items():
        if not isinstance(entry, dict):
            errors.append(f"table manifest entry must be an object: {name}")
            continue
        inputs = entry.get("input_files", [])
        hashes = entry.get("input_sha256", {})
        if not isinstance(hashes, dict):
            errors.append(f"input_sha256 must be an object: {name}")
            continue
        if not isinstance(inputs, list):
            errors.append(f"input_files must be a list: {name}")
            continue
        if not all(isinstance(relative, str) for relative in inputs):
            errors.append(f"input_files entries must be strings: {name}")
            continue
        if set(inputs) != set(hashes):
            errors.append(f"generated asset provenance input list mismatch: {name}")
        for relative, recorded in hashes.items():
            path, path_error = _manifest_input_path(repo_root, relative)
            if path_error:
                errors.append(path_error)
                continue
            assert path is not None
            if not path.is_file() or file_sha256(path) != recorded:
                errors.append(f"generated asset provenance input mismatch: {relative}")
    consistency = run_data.get("consistency")
    if not isinstance(consistency, list) or not consistency:
        errors.append("run manifest consistency gates are missing")
    else:
        invalid = [index for index, check in enumerate(consistency) if not isinstance(check, dict)]
        if invalid:
            errors.append(f"run manifest consistency entry must be an object: {invalid[0]}")
        elif any(check.get("status") != "pass" for check in consistency):
            errors.append("run manifest consistency gates are not all passing")
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
    errors = source_errors(args.root, final=args.final, repo_root=args.repo_root)
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
