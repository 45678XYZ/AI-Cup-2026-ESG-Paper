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
    "tables/table3_legality_cost.tex",
    "tables/table4_contrasts.tex",
    "tables/table5_headroom.tex",
    "tables/table6_regimes.tex",
    "tables/table7_multilingual_mechanism.tex",
    "tables/table8_invalid_anatomy.tex",
    "tables/table9_external_arms.tex",
    "figures/figure1_hierarchy.pdf",
)
MANIFEST_REQUIRED_TABLES = (
    "table1_dataset.tex",
    "table2_main.tex",
    "table4_contrasts.tex",
    "table5_headroom.tex",
    "table6_regimes.tex",
)
CANONICAL_FROZEN_ASSETS = REQUIRED_ASSETS + ("tables/manifest.json",)
REQUIRED_TABLE_INCLUSIONS = (
    ("Table 4", "table4_contrasts.tex"),
    ("Table 7", "table7_multilingual_mechanism.tex"),
    # The evidence tables for the counts the abstract and Introduction repeat.
    # Included by name for the same reason as the two above: a claim whose
    # supporting table drops out of the build is worse than one that never
    # had one, because the prose still asserts it.
    ("Table 8", "table8_invalid_anatomy.tex"),
    ("Table 9", "table9_external_arms.tex"),
)
LAYOUT_PLACEHOLDER_AUTHORS = frozenset(
    f"Student Author {number}" for number in range(1, 5)
)
LAYOUT_PLACEHOLDER_TODO_MARKERS = ("TODO(author-metadata)",)
NTCIR_TRACK_NAME = "AI CUP-VeriPromiseESG Task"
NTCIR_TEAM_NAME = "Team_10537"
NTCIR_SUBTASK = "AI CUP Special Session at NTCIR"
CAPTION_WORD_LIMIT = 20


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


def _main_source(root: Path) -> str:
    path = root / "main.tex"
    try:
        return _active_source(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        return ""


def _has_layout_author_placeholders(metadata: str) -> bool:
    authors = re.findall(r"\\author\{([^}]*)\}", _active_source(metadata))
    return (
        any(author.strip() in LAYOUT_PLACEHOLDER_AUTHORS for author in authors)
        or any(marker in metadata for marker in LAYOUT_PLACEHOLDER_TODO_MARKERS)
    )


def _caption_arguments(text: str) -> list[tuple[int, str]]:
    """Return source offsets and balanced arguments of active caption commands."""
    captions: list[tuple[int, str]] = []
    for match in re.finditer(r"\\caption\s*\{", text):
        start = match.end()
        depth = 1
        position = start
        while position < len(text) and depth:
            if text[position] == "{" and text[position - 1] != "\\":
                depth += 1
            elif text[position] == "}" and text[position - 1] != "\\":
                depth -= 1
            position += 1
        if depth == 0:
            captions.append((match.start(), text[start:position - 1]))
    return captions


def _caption_text(root: Path, argument: str) -> tuple[str, str | None]:
    input_match = re.fullmatch(
        r"\s*(?:\\protect\s*)?\\input\s*\{([^}]+)\}\s*", argument
    )
    if input_match is None:
        return argument, None
    target = input_match.group(1)
    path = (root / target).resolve()
    try:
        return path.read_text(encoding="utf-8"), str(path)
    except (OSError, UnicodeError):
        return argument, None


def _english_word_count(caption: str) -> int:
    rendered = re.sub(r"\$[^$]*\$", " ", caption)
    rendered = re.sub(r"\\(?:ref|pageref|label)\s*\{[^{}]*\}", " ", rendered)
    rendered = re.sub(r"\\[A-Za-z@]+\*?", " ", rendered)
    rendered = re.sub(r"\\.", " ", rendered)
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z0-9]+)*", rendered))


def caption_errors(root: Path) -> list[str]:
    errors: list[str] = []
    paths = sorted({
        path for pattern in ("*.tex", "sections/*.tex")
        for path in root.glob(pattern) if path.is_file()
    })
    for path in paths:
        try:
            active = _active_source(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            continue
        for offset, argument in _caption_arguments(active):
            caption, imported_from = _caption_text(root, argument)
            words = _english_word_count(caption)
            if words <= CAPTION_WORD_LIMIT:
                continue
            source = imported_from or f"{path}:{active.count(chr(10), 0, offset) + 1}"
            errors.append(
                f"caption exceeds {CAPTION_WORD_LIMIT} English words: "
                f"{source} has {words}"
            )
    return errors


def source_errors(root: Path, final: bool = False, repo_root: Path | None = None) -> list[str]:
    text = source_text(root)
    active = _active_source(text)
    lowered = active.lower()
    errors: list[str] = []
    main = _main_source(root)
    title_start = main.find(r"\title")
    title_end_candidates = (
        main.find(r"\begin{abstract}", title_start),
        main.find(r"\keywords", title_start),
        main.find(r"\maketitle", title_start),
    )
    title_end = min(position for position in title_end_candidates if position >= 0) \
        if any(position >= 0 for position in title_end_candidates) else -1
    title = main[title_start:title_end] if title_start >= 0 and title_end >= 0 else ""
    if NTCIR_TRACK_NAME not in title:
        errors.append(f"paper title must include {NTCIR_TRACK_NAME}")

    keyword_pos = main.find(r"\keywords")
    maketitle_pos = main.find(r"\maketitle")
    team_pos = main.find(r"\section*{Team Name}")
    team_value_pos = main.find(r"\teamname", team_pos + 1) if team_pos >= 0 else -1
    subtasks_pos = main.find(r"\section*{Subtasks}")
    subtask_value_pos = main.find(r"\subtasks", subtasks_pos + 1) \
        if subtasks_pos >= 0 else -1
    introduction_positions = (
        main.find(r"\input{sections/01_introduction}"),
        main.find(r"\section{Introduction}"),
    )
    introduction_pos = min(position for position in introduction_positions if position >= 0) \
        if any(position >= 0 for position in introduction_positions) else -1
    if team_pos < 0:
        errors.append("paper must include a Team Name field")
    if subtasks_pos < 0:
        errors.append("paper must include a Subtasks field")
    if not (
        keyword_pos >= 0
        and keyword_pos < maketitle_pos < team_pos < team_value_pos
        < subtasks_pos < subtask_value_pos < introduction_pos
    ):
        errors.append(
            "required NTCIR front-matter sequence is Keywords, maketitle, "
            "Team Name value, Subtasks value, Introduction"
        )
    active_metadata = _active_source(_metadata(root))
    team_match = re.search(
        r"\\newcommand\s*\{\\teamname\}\s*\{([^{}]*)\}", active_metadata, re.DOTALL
    )
    team_name = " ".join(team_match.group(1).split()).replace(r"\_", "_") \
        if team_match else ""
    if team_name != NTCIR_TEAM_NAME:
        errors.append(f"Team Name must be {NTCIR_TEAM_NAME}")
    subtask_match = re.search(
        r"\\newcommand\s*\{\\subtasks\}\s*\{([^{}]*)\}", active_metadata, re.DOTALL
    )
    subtask = " ".join(subtask_match.group(1).split()) if subtask_match else ""
    if subtask != NTCIR_SUBTASK:
        errors.append(f"Subtasks must be {NTCIR_SUBTASK}")
    errors.extend(caption_errors(root))
    if any(marker.lower() in lowered for marker in PROHIBITED_HARDWARE):
        errors.append("manuscript contains a prohibited hardware reference")
    if re.search(r"\bno difference\b", lowered):
        errors.append("replace 'no difference' with 'no detectable difference'")
    if re.search(r"\bequivalence\b|\bequivalent\b", lowered):
        errors.append("an equivalence claim requires an equivalence design")
    if re.search(r"\bm7\b", lowered):
        errors.append("M7 is outside the focused manuscript")
    inclusions = re.findall(r"\\(?:input|include)\s*\{([^}]+)\}", active)
    for table_label, table_name in REQUIRED_TABLE_INCLUSIONS:
        targets = [target for target in inclusions if Path(target).name == table_name]
        if not targets:
            errors.append(f"the full generated {table_name} is not included")
            continue
        canonical = ((repo_root or root) / "tables" / table_name).resolve()
        for target in targets:
            resolved = (root / target).resolve()
            if resolved != canonical:
                errors.append(
                    f"{table_label} inclusion must resolve to canonical generated asset: "
                    f"{target}"
                )
            elif not resolved.is_file():
                errors.append(f"included generated asset is missing: {target}")
    if any(Path(target).name == "table3_regimes.tex" for target in inclusions):
        errors.append("the selected-best Table 3 is excluded from the focused manuscript")
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
                ["git", "diff", "--quiet", "HEAD", "--", relative],
                cwd=repo_root, capture_output=True, check=False,
            ).returncode == 0
        except OSError:
            tracked, clean = False, False
        if not tracked:
            errors.append(f"canonical generated asset is not tracked: {relative}")
        elif not clean:
            errors.append(f"canonical generated asset is not clean relative to HEAD: {relative}")
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
    required_tables = MANIFEST_REQUIRED_TABLES
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
    overfull_pattern = r"overfull \\(hbox|vbox) \(([\d.]+)pt too (wide|high)\)"
    for box_type, excess, direction in re.findall(overfull_pattern, text):
        if float(excess) > 2.0:
            errors.append(
                f"build log has an overfull {box_type} {excess}pt too {direction}"
            )
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
