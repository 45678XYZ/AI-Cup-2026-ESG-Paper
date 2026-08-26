import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf._text_extraction import mult
from paper.data import REPO_ROOT

from manuscript.check import (
    asset_errors,
    font_errors,
    log_errors,
    pdf_errors,
    source_errors,
    source_warnings,
    source_text,
    main,
)


CANONICAL_TABLE_NAMES = (
    "table1_dataset.tex",
    "table2_main.tex",
    "table3_legality_cost.tex",
    "table4_contrasts.tex",
    "table5_headroom.tex",
    "table6_regimes.tex",
    "table7_multilingual_mechanism.tex",
)


def write_minimal(root: Path, body: str, metadata: str = "") -> None:
    root.mkdir()
    (root / "main.tex").write_text(body, encoding="utf-8")
    (root / "metadata.tex").write_text(metadata, encoding="utf-8")


def write_policy_valid(root: Path, metadata: str = "") -> None:
    write_minimal(
        root,
        "\\input{tables/table4_contrasts.tex}\n"
        "\\input{tables/table7_multilingual_mechanism.tex}",
        metadata,
    )
    (root / "tables").mkdir()
    for name in ("table4_contrasts.tex", "table7_multilingual_mechanism.tex"):
        (root / "tables" / name).write_text("", encoding="utf-8")


def make_valid_asset_repo(root: Path) -> Path:
    """Create a committed main branch that satisfies every asset policy."""
    from paper.data import file_sha256

    tables = root / "tables"
    figures = root / "figures"
    analysis = root / "analysis"
    tables.mkdir(parents=True)
    figures.mkdir()
    analysis.mkdir()
    source = root / "input.json"
    source.write_text("original input\n", encoding="utf-8")
    (analysis / "generate.py").write_text("# fixture generator\n", encoding="utf-8")
    for name in CANONICAL_TABLE_NAMES:
        (tables / name).write_text(f"{name}\n", encoding="utf-8")
    (figures / "figure1_hierarchy.pdf").write_bytes(b"fixture pdf")
    table_entry = {
        "source_script": "analysis/generate.py",
        "input_files": ["input.json"],
        "input_sha256": {"input.json": file_sha256(source)},
    }
    (tables / "manifest.json").write_text(
        json.dumps({"tables": {name: table_entry for name in CANONICAL_TABLE_NAMES}}),
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text(
        json.dumps({"consistency": [{"status": "pass"}]}),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-qb", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.org"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root


def test_draft_rejects_prohibited_hardware_and_unqualified_null_claim(tmp_path):
    write_minimal(tmp_path / "m", "\\input{../tables/table4_contrasts.tex}\n"
                  "Path-constrained wF1 and hierarchical F1 are post hoc. "
                  "There was no difference on an L" + "40S run.")
    errors = source_errors(tmp_path / "m")
    assert any("prohibited hardware" in error for error in errors)
    assert any("no detectable difference" in error for error in errors)


def test_draft_rejects_an_equivalence_claim(tmp_path):
    write_minimal(tmp_path / "m", "\\input{../tables/table4_contrasts.tex}\n"
                  "Path-constrained wF1 and hierarchical F1 are post hoc. "
                  "The two decision rules are equivalent.")
    assert any("equivalence" in error for error in source_errors(tmp_path / "m"))


def test_draft_requires_full_table4_and_post_hoc_disclosure(tmp_path):
    write_minimal(tmp_path / "m", "Path-constrained wF1 and hierarchical F1.")
    errors = source_errors(tmp_path / "m")
    assert any("table4_contrasts.tex" in error for error in errors)
    assert any("post hoc" in error for error in errors)


def test_author_metadata_is_draft_warning_but_final_error(tmp_path):
    write_minimal(tmp_path / "m", "\\input{../tables/table4_contrasts.tex}\n"
                  "Path-constrained wF1 and hierarchical F1 were adopted post hoc.")
    assert not any("author metadata" in error for error in source_errors(tmp_path / "m"))
    assert any("author metadata" in warning for warning in source_warnings(tmp_path / "m"))
    assert any("author metadata" in error for error in source_errors(tmp_path / "m", final=True))


def test_layout_author_placeholders_warn_in_draft_and_block_final_even_with_email(tmp_path):
    root = tmp_path / "m"
    metadata = "\n".join(
        [*(f"\\author{{Student Author {number}}}" for number in range(1, 5)),
         "\\email{student@example.org}"]
    )
    write_policy_valid(root, metadata)

    assert source_errors(root) == []
    assert any("layout-only author placeholders" in warning for warning in source_warnings(root))
    assert any("layout-only author placeholders" in error
               for error in source_errors(root, final=True))


def test_real_author_and_email_metadata_are_accepted_in_final_mode(tmp_path):
    root = tmp_path / "m"
    write_policy_valid(
        root, "\\author{Ada Example}\n\\email{ada@example.org}"
    )

    assert source_errors(root, final=True) == []
    assert source_warnings(root) == []


def test_commented_author_and_email_do_not_satisfy_final_metadata(tmp_path):
    root = tmp_path / "m"
    write_minimal(
        root,
        "\\input{tables/table4_contrasts.tex}",
        "% \\author{Commented Example}\n% \\email{commented@example.org}",
    )
    (root / "tables").mkdir()
    (root / "tables" / "table4_contrasts.tex").write_text("", encoding="utf-8")

    assert any("author metadata" in error for error in source_errors(root, final=True))
    assert any("author metadata" in warning for warning in source_warnings(root))


def test_commented_layout_placeholders_do_not_block_active_real_metadata(tmp_path):
    root = tmp_path / "m"
    commented_placeholders = "\n".join(
        f"% \\author{{Student Author {number}}}" for number in range(1, 5)
    )
    write_policy_valid(
        root,
        f"{commented_placeholders}\n\\author{{Ada Example}}\n\\email{{ada@example.org}}",
    )

    assert source_errors(root, final=True) == []
    assert source_warnings(root) == []


def test_explicit_author_metadata_todo_marker_warns_and_blocks_final_mode(tmp_path):
    root = tmp_path / "m"
    write_minimal(
        root,
        "\\input{tables/table4_contrasts.tex}",
        "% TODO(author-metadata)\n\\author{Ada Example}\n\\email{ada@example.org}",
    )
    (root / "tables").mkdir()
    (root / "tables" / "table4_contrasts.tex").write_text("", encoding="utf-8")

    assert any("layout-only author placeholders" in warning for warning in source_warnings(root))
    assert any("layout-only author placeholders" in error
               for error in source_errors(root, final=True))


def test_pdf_page_limit(tmp_path):
    path = tmp_path / "nine-pages.pdf"
    writer = PdfWriter()
    for _ in range(9):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as stream:
        writer.write(stream)
    assert any("9 pages" in error for error in pdf_errors(path, max_pages=8))


def test_tracked_manuscript_omits_snapshot_revision():
    reader = PdfReader(str(REPO_ROOT / "manuscript" / "build" / "main.pdf"))
    document_text = "\n".join(page.extract_text() for page in reader.pages)

    assert "snapshot revision" not in document_text
    assert "a25cc9e05974bd9687e528edd516f2cfdb3f5db9" not in document_text


def test_tracked_manuscript_has_no_empty_body_column():
    """A text-heavy body page must not abandon either ACM column."""
    reader = PdfReader(str(REPO_ROOT / "manuscript" / "build" / "main.pdf"))
    imbalanced = []
    for page_number, page in enumerate(reader.pages[:-1], start=1):
        midpoint = float(page.mediabox.width) / 2
        words = [0, 0]

        def count_words(text, cm, tm, font_dict, font_size):
            clean = text.split()
            if clean:
                x = mult(tm, cm)[4]
                words[0 if x < midpoint else 1] += len(clean)

        page.extract_text(visitor_text=count_words)
        total = sum(words)
        if total >= 200 and min(words) < total * 0.1:
            imbalanced.append((page_number, words))

    assert imbalanced == [], f"body pages with an effectively empty column: {imbalanced}"


def test_tracked_manuscript_title_is_compact_and_balanced():
    """The rendered title must not be split into narrow manual fragments."""
    page = PdfReader(
        str(REPO_ROOT / "manuscript" / "build" / "main.pdf")
    ).pages[0]
    midpoint = float(page.mediabox.width) / 2
    title_lines = []

    def collect_title_lines(text, cm, tm, font_dict, font_size):
        clean = " ".join(text.split())
        if clean and font_size >= 16:
            x = mult(tm, cm)[4]
            title_lines.append((clean, 2 * (midpoint - x)))

    page.extract_text(visitor_text=collect_title_lines)
    expected_title = (
        "Field-Wise Metrics Can Miss Hierarchical Gains: "
        "Multilingual Evidence from ESG Promise Verification"
    )
    widths = [width for _, width in title_lines]

    assert " ".join(text for text, _ in title_lines) == expected_title
    assert len(title_lines) <= 3, f"title rendered across {len(title_lines)} lines"
    assert min(widths) >= max(widths) * 0.65, f"unbalanced title widths: {widths}"


def test_log_rejects_unresolved_references(tmp_path):
    log = tmp_path / "main.log"
    log.write_text("LaTeX Warning: There were undefined references.", encoding="utf-8")
    assert log_errors(log)


def test_asset_check_names_missing_generated_files(tmp_path):
    errors = asset_errors(tmp_path)
    assert any("table4_contrasts.tex" in error for error in errors)


def test_source_text_reads_root_sections_and_bib(tmp_path):
    (tmp_path / "main.tex").write_text("root", encoding="utf-8")
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "method.tex").write_text("section", encoding="utf-8")
    (tmp_path / "refs.bib").write_text("bib", encoding="utf-8")
    assert source_text(tmp_path) == "root\nbib\nsection"


def _real_abstract() -> str:
    main = (REPO_ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.DOTALL)
    assert match is not None
    return match.group(1)


def test_focused_manuscript_states_the_audit_and_replication_contract():
    text = source_text(REPO_ROOT / "manuscript")
    for claim in (
        "49 companies",
        "32/32",
        "14/32",
        "AI CUP weights applied to ML-Promise",
    ):
        assert claim in text
    assert "(not official)" not in text


def test_focused_abstract_prioritizes_prespecified_tuple_evidence():
    abstract = _real_abstract()
    assert re.search(
        r"\+0\.035.*?p_\{\\mathrm\{Holm\}\}=\.001", abstract, re.DOTALL
    )
    assert r"p_{\mathrm{Holm}}=.025" not in abstract


def test_focused_manuscript_does_not_rank_languages_by_raw_score():
    text = source_text(REPO_ROOT / "manuscript").lower()
    for prohibited in (
        "best language",
        "highest-scoring language",
        "japanese is harder",
        "korean is easier",
        "outperforms the other languages",
    ):
        assert prohibited not in text


def test_equivalence_language_and_bound_metric_names_require_disclosure(tmp_path):
    write_minimal(tmp_path / "m", "C-wF1 and hF establish equivalence.")
    errors = source_errors(tmp_path / "m")
    assert any("equivalence" in error for error in errors)
    assert any("post hoc" in error for error in errors)


def test_table4_comment_spoof_is_not_an_inclusion(tmp_path):
    write_minimal(tmp_path / "m", "% \\input{../tables/table4_contrasts.tex}")
    assert any("table4_contrasts.tex" in error for error in source_errors(tmp_path / "m"))


def test_draft_requires_the_canonical_multilingual_table(tmp_path):
    repo = tmp_path / "repo"
    manuscript = repo / "manuscript"
    tables = repo / "tables"
    tables.mkdir(parents=True)
    (tables / "table4_contrasts.tex").write_text("table 4\n", encoding="utf-8")
    (tables / "table7_multilingual_mechanism.tex").write_text(
        "table 7\n", encoding="utf-8"
    )
    write_minimal(manuscript, "\\input{../tables/table4_contrasts.tex}")

    errors = source_errors(manuscript, repo_root=repo)
    assert any("table7_multilingual_mechanism.tex" in error for error in errors)


def test_draft_rejects_an_external_multilingual_table_decoy(tmp_path):
    repo = tmp_path / "repo"
    manuscript = repo / "manuscript"
    tables = repo / "tables"
    external = tmp_path / "external" / "table7_multilingual_mechanism.tex"
    tables.mkdir(parents=True)
    external.parent.mkdir()
    (tables / "table4_contrasts.tex").write_text("table 4\n", encoding="utf-8")
    (tables / "table7_multilingual_mechanism.tex").write_text(
        "canonical\n", encoding="utf-8"
    )
    external.write_text("decoy\n", encoding="utf-8")
    write_minimal(
        manuscript,
        "\\input{../tables/table4_contrasts.tex}\n"
        "\\input{../../external/table7_multilingual_mechanism.tex}",
    )

    errors = source_errors(manuscript, repo_root=repo)
    assert any("Table 7 inclusion must resolve" in error for error in errors)


def test_draft_rejects_active_m7_but_not_a_comment(tmp_path):
    active = tmp_path / "active"
    commented = tmp_path / "commented"
    write_minimal(active, "We evaluate M7.")
    write_minimal(commented, "% M7 was removed.")

    assert any("M7" in error for error in source_errors(active))
    assert not any("M7" in error for error in source_errors(commented))


def test_draft_rejects_the_selected_best_regime_table(tmp_path):
    root = tmp_path / "m"
    write_minimal(root, "\\input{../tables/table3_regimes.tex}")

    assert any("selected-best Table 3" in error for error in source_errors(root))


def test_cli_rejects_external_table4_decoy(tmp_path, monkeypatch, capsys):
    repo = make_valid_asset_repo(tmp_path)
    manuscript = repo / "manuscript"
    external = repo.parent / f"{repo.name}-external" / "table4_contrasts.tex"
    external.parent.mkdir()
    external.write_text("decoy\n", encoding="utf-8")
    write_minimal(manuscript, f"\\input{{../../{external.parent.name}/table4_contrasts.tex}}")
    monkeypatch.setattr(
        "sys.argv",
        ["check", "--root", str(manuscript), "--repo-root", str(repo)],
    )

    assert main() == 1
    assert "Table 4 inclusion must resolve to canonical generated asset" in capsys.readouterr().out


def test_asset_check_rejects_noncontained_manifest_inputs(tmp_path):
    from paper.data import file_sha256

    cases = ("absolute", "traversal", "external symlink")
    for case in cases:
        repo = make_valid_asset_repo(tmp_path / case.replace(" ", "-"))
        external = repo.parent / f"{repo.name}-external.json"
        external.write_text("original input\n", encoding="utf-8")
        if case == "absolute":
            manifest_input = str(external)
            expected = "must be repository-relative"
        elif case == "traversal":
            manifest_input = f"../{external.name}"
            expected = "must not traverse parent directories"
        else:
            link = repo / "linked-input.json"
            link.symlink_to(external)
            manifest_input = link.name
            expected = "resolves outside repository"
        manifest_path = repo / "tables" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["tables"]["table1_dataset.tex"]
        entry["input_files"] = [manifest_input]
        entry["input_sha256"] = {manifest_input: file_sha256(external)}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", case], cwd=repo, check=True)

        errors = asset_errors(repo)
        assert any(expected in error for error in errors), (case, errors)


@pytest.mark.parametrize("staged", [False, True], ids=["unstaged", "staged"])
def test_asset_check_detects_tampered_canonical_table(tmp_path, staged):
    repo = make_valid_asset_repo(tmp_path)
    assert asset_errors(repo) == []

    (repo / "tables" / "table4_contrasts.tex").write_text("tampered\n", encoding="utf-8")
    if staged:
        subprocess.run(
            ["git", "add", "tables/table4_contrasts.tex"], cwd=repo, check=True
        )

    assert "canonical generated asset is not clean relative to HEAD: " \
           "tables/table4_contrasts.tex" in asset_errors(repo)


def test_asset_check_accepts_a_committed_table_newer_than_main(tmp_path):
    repo = make_valid_asset_repo(tmp_path)
    subprocess.run(["git", "switch", "-qc", "paper"], cwd=repo, check=True)
    (repo / "tables" / "table1_dataset.tex").write_text(
        "corrected table\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "tables/table1_dataset.tex"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "correct audit"], cwd=repo, check=True)

    assert asset_errors(repo) == []


def test_asset_check_detects_tampered_table_manifest(tmp_path):
    repo = make_valid_asset_repo(tmp_path)
    assert asset_errors(repo) == []

    manifest = repo / "tables" / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert "canonical generated asset is not clean relative to HEAD: " \
           "tables/manifest.json" in asset_errors(repo)


def test_asset_check_rejects_untracked_canonical_table(tmp_path):
    repo = make_valid_asset_repo(tmp_path)
    assert asset_errors(repo) == []
    subprocess.run(
        ["git", "rm", "--cached", "tables/table3_legality_cost.tex"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    assert "canonical generated asset is not tracked: tables/table3_legality_cost.tex" \
           in asset_errors(repo)


def test_font_checker_accepts_pdf_without_font_resources(tmp_path):
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as stream:
        writer.write(stream)
    assert font_errors(path) == []


def test_clean_compiled_manuscript_renders_without_system_fonts(tmp_path):
    """The archived source must build without host-installed fonts."""
    manuscript = REPO_ROOT / "manuscript"
    empty_fonts = tmp_path / "empty-fonts"
    empty_fonts.mkdir()
    font_cache = tmp_path / "font-cache"
    font_config = tmp_path / "fonts.conf"
    font_config.write_text(
        "<?xml version='1.0'?>\n"
        "<!DOCTYPE fontconfig SYSTEM 'urn:fontconfig:fonts.dtd'>\n"
        "<fontconfig>\n"
        f"  <dir>{empty_fonts.as_posix()}</dir>\n"
        f"  <cachedir>{font_cache.as_posix()}</cachedir>\n"
        "</fontconfig>\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["FONTCONFIG_FILE"] = str(font_config)
    env["FONTCONFIG_PATH"] = str(tmp_path)
    subprocess.run(["make", "clean"], cwd=manuscript, check=True, capture_output=True, text=True)
    subprocess.run(
        ["make", "check"], cwd=manuscript, env=env, check=True,
        capture_output=True, text=True,
    )

    pdf = manuscript / "build" / "main.pdf"
    reader = PdfReader(str(pdf))
    page_text = [page.extract_text().strip() for page in reader.pages]
    document_text = "\n".join(page_text)
    for caption in (
        "AI CUP 2026 dataset summary.",
        "Document-disjoint cross-fitted results on the Chinese development set.",
        "Projection (M1) against independent argmax (M0) on five corpora",
        "Chinese paired contrasts on four metrics.",
    ):
        assert caption in document_text
    assert all(page_text), "the compiled manuscript contains a blank page"
    sparse_body_pages = [
        index for index, text in enumerate(page_text[:-1], start=1)
        if len(text.split()) < 100
    ]
    assert not sparse_body_pages, (
        f"the compiled manuscript contains near-empty body pages: {sparse_body_pages}"
    )
    table4_caption = "Chinese paired contrasts on four metrics."
    external_start = "Across all four external corpora"
    discussion_heading = "7 DISCUSSION"
    assert document_text.index(table4_caption) < document_text.index(discussion_heading)
    assert document_text.index(external_start) < document_text.index(discussion_heading)
    normalized_document_text = " ".join(document_text.split())
    assert "AI CUP 2026 競賽提交格式說明 Sample Submission Format Guide" not in (
        normalized_document_text
    )
    assert (
        "SemEval-2025 Task 6: Multinational, Multilingual, Multi-Industry "
        "Promise Verification"
    ) in normalized_document_text
    metadata = (manuscript / "metadata.tex").read_text(encoding="utf-8")
    authors = re.findall(r"\\author\{([^}]+)\}", metadata)
    emails = re.findall(r"\\email\{([^}]+)\}", metadata)
    assert len(authors) == 4
    assert len(emails) == 4
    assert not any(author.startswith("Student Author ") for author in authors)
    assert all(author in document_text for author in authors)
    assert all(email in document_text for email in emails)
    assert font_errors(pdf) == []
    log = (manuscript / "build" / "main.log").read_text(encoding="utf-8")
    assert "accessing absolute path" not in log
    assert "Missing character" not in log
    assert "could not represent character" not in log


def test_cli_returns_nonzero_for_policy_error(tmp_path, monkeypatch, capsys):
    write_minimal(tmp_path / "m", "no difference")
    monkeypatch.setattr("sys.argv", ["check", "--root", str(tmp_path / "m"), "--repo-root", str(tmp_path)])
    assert main() == 1
    assert "ERROR:" in capsys.readouterr().out


def test_canonical_repository_assets_have_clean_provenance():
    assert asset_errors(REPO_ROOT) == []


def test_missing_required_table_manifest_entry_is_rejected(tmp_path):
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    tables.mkdir(); figures.mkdir()
    for name in CANONICAL_TABLE_NAMES:
        (tables / name).write_text(name, encoding="utf-8")
    (figures / "figure1_hierarchy.pdf").write_bytes(b"pdf")
    (tables / "manifest.json").write_text(json.dumps({"tables": {}}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"consistency": []}), encoding="utf-8")
    errors = asset_errors(tmp_path)
    assert any("manifest entry" in error for error in errors)


def test_tampered_table_input_is_rejected(tmp_path):
    tables = tmp_path / "tables"; tables.mkdir()
    source = tmp_path / "input.json"; source.write_text("original", encoding="utf-8")
    (tables / "manifest.json").write_text(json.dumps({"tables": {
        name: {"source_script": "script.py", "input_files": ["input.json"],
               "input_sha256": {"input.json": "sha256:bad"}}
        for name in CANONICAL_TABLE_NAMES
    }}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"consistency": []}), encoding="utf-8")
    errors = asset_errors(tmp_path)
    assert any("input mismatch" in error for error in errors)


def test_missing_or_failed_run_manifest_consistency_is_rejected(tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "manifest.json").write_text(json.dumps({"tables": {}}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"consistency": [{"status": "fail"}]}), encoding="utf-8")
    assert any("consistency" in error for error in asset_errors(tmp_path))


def test_untracked_figure_is_rejected(tmp_path):
    (tmp_path / "tables").mkdir(); (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "figure1_hierarchy.pdf").write_bytes(b"pdf")
    assert any("figure" in error for error in asset_errors(tmp_path))


def test_asset_check_rejects_malformed_but_valid_manifest_shapes(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    (tmp_path / "run_manifest.json").write_text(json.dumps({"consistency": []}), encoding="utf-8")
    cases = [
        ([], "manifest top level"),
        ({"tables": []}, "tables section"),
        ({"tables": {"table1_dataset.tex": []}}, "table manifest entry"),
        ({"tables": {"table1_dataset.tex": {"input_sha256": []}}}, "input_sha256"),
    ]
    for index, (payload, label) in enumerate(cases):
        path = tables / f"manifest-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.rename(tables / "manifest.json")
        errors = asset_errors(tmp_path)
        assert any(label in error for error in errors), (label, errors)


def test_asset_check_rejects_malformed_consistency_shapes(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "manifest.json").write_text(json.dumps({"tables": {}}), encoding="utf-8")
    for payload in ({"consistency": {}}, {"consistency": [{}]}, {"consistency": ["bad"]}):
        (tmp_path / "run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        errors = asset_errors(tmp_path)
        assert any("consistency" in error for error in errors)


def test_asset_check_rejects_malformed_input_file_shapes(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    names = CANONICAL_TABLE_NAMES
    for name in names:
        (tables / name).write_text(name, encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"consistency": [{"status": "pass"}]}), encoding="utf-8"
    )
    malformed_inputs = (
        "input.json",
        [[]],
        [{}],
        [0],
        [None],
        [True],
        [False],
    )
    for inputs in malformed_inputs:
        manifest = {"tables": {
            name: {
                "source_script": "script.py",
                "input_files": inputs,
                "input_sha256": {},
            }
            for name in names
        }}
        (tables / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        errors = asset_errors(tmp_path)
        assert any("input_files" in error for error in errors), (inputs, errors)
