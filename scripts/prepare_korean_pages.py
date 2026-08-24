"""Derive the Korean model input from the report pages named by ML-Promise.

The released ``Trainset_Korean.json`` has one label tuple per unique
``(URL, page_number)`` but no text field.  This script downloads each report
once, extracts only the labelled pages with Poppler's ``pdftotext``, and writes
a local augmented JSON.  Reports and extracted text are deliberately ignored
by Git; only the extraction recipe and its manifest format are versioned.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "dataset" / "mlpromise_korean.json"
LOCAL_ROOT = REPO_ROOT / "local_data"
PDF_ROOT = LOCAL_ROOT / "mlpromise_korean_pdfs"
OUTPUT = LOCAL_ROOT / "mlpromise_korean_pages.json"
MANIFEST = LOCAL_ROOT / "mlpromise_korean_pages_manifest.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def pdf_path(url: str) -> Path:
    return PDF_ROOT / (hashlib.sha256(url.encode("utf-8")).hexdigest()[:20] + ".pdf")


def download(url: str, attempts: int = 3) -> Path:
    target = pdf_path(url)
    if target.exists() and target.read_bytes()[:4] == b"%PDF":
        return target
    part = target.with_suffix(".pdf.part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 ML-Promise-replication/1.0"},
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=180) as response, open(part, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if part.read_bytes()[:4] != b"%PDF":
                raise ValueError(f"downloaded response is not a PDF: {url}")
            os.replace(part, target)
            return target
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def extract_page(path: Path, page_number: int) -> str:
    completed = subprocess.run(
        ["pdftotext", "-f", str(page_number), "-l", str(page_number),
         "-layout", str(path), "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.decode("utf-8", errors="replace").strip()


def ocr_page(path: Path, page_number: int, tesseract: str) -> str:
    """OCR one image-only page; used only when pdftotext returns nothing."""
    with tempfile.TemporaryDirectory(prefix="mlpromise-ko-ocr-") as temporary:
        prefix = Path(temporary) / "page"
        subprocess.run(
            ["pdftoppm", "-f", str(page_number), "-l", str(page_number),
             "-r", "250", "-png", "-singlefile", str(path), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        completed = subprocess.run(
            [tesseract, str(prefix) + ".png", "stdout", "-l", "kor+eng",
             "--psm", "3"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return completed.stdout.decode("utf-8", errors="replace").strip()


def write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(temporary, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel report downloads (default: 4)")
    ap.add_argument(
        "--tesseract",
        default=None,
        help="tesseract executable for image-only pages; defaults to the "
             "local OCR environment and then PATH",
    )
    args = ap.parse_args()

    local_tesseract = LOCAL_ROOT / "ocr-env" / "bin" / "tesseract"
    tesseract = args.tesseract or (
        str(local_tesseract) if local_tesseract.exists() else shutil.which("tesseract")
    )

    with open(SOURCE, encoding="utf-8") as f:
        rows = json.load(f)
    urls = sorted({str(row["URL"]).strip() for row in rows})
    if len(rows) != 500 or len(urls) != 32:
        raise SystemExit(
            f"Korean release shape changed: {len(rows)} rows, {len(urls)} reports"
        )
    url_pages = [(str(row["URL"]).strip(), int(row["page_number"])) for row in rows]
    if len(set(url_pages)) != len(rows):
        raise SystemExit("Korean release no longer has one row per report page")

    PDF_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"downloading/reusing {len(urls)} reports in {PDF_ROOT}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_url = {pool.submit(download, url): url for url in urls}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url), 1):
            url = future_to_url[future]
            path = future.result()
            print(f"[{i:02d}/{len(urls)}] {path.name} {path.stat().st_size / 1e6:.1f} MB", flush=True)

    augmented = []
    empty = []
    ocr_rows = []
    for i, row in enumerate(rows):
        text = extract_page(pdf_path(str(row["URL"]).strip()), int(row["page_number"]))
        if not text:
            if tesseract:
                text = ocr_page(
                    pdf_path(str(row["URL"]).strip()),
                    int(row["page_number"]),
                    tesseract,
                )
                ocr_rows.append(i)
        out = dict(row)
        out["data"] = text
        augmented.append(out)
        if not text:
            empty.append(i)
    write_json_atomic(OUTPUT, augmented)

    version = subprocess.run(
        ["pdftotext", "-v"], text=True, capture_output=True, check=True,
    ).stderr.splitlines()[0]
    tesseract_version = None
    if tesseract:
        tesseract_version = subprocess.run(
            [tesseract, "--version"], text=True, capture_output=True, check=True,
        ).stdout.splitlines()[0]
    reports = {
        url: {
            "path": str(pdf_path(url).relative_to(REPO_ROOT)),
            "sha256": file_sha256(pdf_path(url)),
            "bytes": pdf_path(url).stat().st_size,
        }
        for url in urls
    }
    manifest = {
        "source": str(SOURCE.relative_to(REPO_ROOT)),
        "source_sha256": file_sha256(SOURCE),
        "output": str(OUTPUT.relative_to(REPO_ROOT)),
        "output_sha256": file_sha256(OUTPUT),
        "method": "pdftotext -f PAGE -l PAGE -layout PDF -",
        "pdftotext_version": version,
        "ocr_fallback": "pdftoppm -r 250 followed by tesseract -l kor+eng --psm 3",
        "tesseract_version": tesseract_version,
        "ocr_row_indexes": ocr_rows,
        "n_rows": len(augmented),
        "n_reports": len(reports),
        "empty_text_row_indexes": empty,
        "reports": reports,
    }
    write_json_atomic(MANIFEST, manifest)
    print(
        f"wrote {OUTPUT} ({len(augmented)} pages; {len(empty)} empty)\n"
        f"manifest {MANIFEST}",
        flush=True,
    )
    if empty:
        raise SystemExit(
            "Some labelled pages have no extractable text. Do not train Korean "
            "until OCR or an author-supplied text release resolves them: "
            + ", ".join(map(str, empty))
        )


if __name__ == "__main__":
    main()
