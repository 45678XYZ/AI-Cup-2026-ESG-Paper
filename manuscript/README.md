# NTCIR-19 manuscript

This directory contains the LaTeX source for the AI CUP ESG manuscript. It
inputs frozen generated tables and the committed hierarchy figure from the
repository root; do not edit those artifacts while drafting the paper.

Run `make build` to compile `build/main.pdf`. Run `make check` for draft
policy, provenance, page-count, log, and font checks. `make check-final` also
requires author metadata, so it is expected to fail until the authors provide
their submission details.
