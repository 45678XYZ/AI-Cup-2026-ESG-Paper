# NTCIR-19 manuscript

This directory contains the LaTeX source for the AI CUP ESG manuscript. It
inputs frozen generated tables and the committed hierarchy figure from the
repository root; do not edit those artifacts while drafting the paper.

Run `make build` to compile `build/main.pdf`. Run `make check` for draft
policy, provenance, page-count, log, and font checks. `make check-final` also
requires real author metadata, so it is expected to fail while the four visible
`Student Author 1`--`Student Author 4` layout-only placeholders remain. Replace
all four with real names and add user-supplied affiliation and email details
before final submission; do not invent any of those details.
