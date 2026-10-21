# Poster layout

## Canvas

- A0 portrait, 841 × 1189 mm with 20 mm margins, built from `article` with `tcolorbox` blocks.
- Two-column rows use `tcbraster` with `raster equal height=rows`, so both blocks in a row end on the same line. Full-width rows hold Figure 1 and the result tables.
- Do not leave blank lines between blocks inside a `tcbraster`: a blank line starts a new row, and the blocks then stack instead of sitting side by side.

## Grid (top to bottom)

| Row | Left | Right |
|---|---|---|
| Header band (full width) | NCU logo · title, authors, affiliation, team, venue · NTCIR logo | — |
| A | **1 · Introduction**, ending in the research question below a thin rule | **2 · Task & Data** |
| B (full width) | **3 · Methodology**: Figure 1 (50%) and two lists, output rules and evaluation (46%) | — |
| C (full width) | **4 · Results: Chinese Anchor**: Table 2 (57.5%) and Table 4 (39.5%), then two bullets | — |
| D | **5 · Multilingual Replication**: Table 7 and two bullets | **6 · Why the Metrics Disagree** |
| E | **7 · Implications for Audit Use** | **8 · Conclusion**, ending in the takeaway below a thin rule |
| Footer band (full width, navy) | References [1]–[4], one per line · QR code on a white square | — |

## Color

One blue family, so the navy footer band balances the navy header and the poster is not top-heavy. Green appears only in the NTCIR logo's leaf.

| Token | RGB | Use |
|---|---|---|
| navy | 12, 45, 87 | Header and footer bands, block title bars, bullets, sub-headings |
| accent | 26, 99, 176 | The five headline results only: 12.55% → 0, +0.035, −0.006, 32/32, 14/32 |
| numeral | 150, 184, 224 | Section numbers and separators on navy |
| rule | 206, 216, 228 | Block frames and the thin rules above the research question and takeaway |
| ink | 28, 36, 46 | Body text |
| muted | 88, 100, 116 | Table and figure notes |

No callout boxes, no colored left borders, no rounded corners: these are the most common marks of generated layouts. Other key numbers are plain bold.

## Type

- Fira Sans throughout, set ragged-right with hyphenation off.
- Title 76 pt bold, authors 40 pt, affiliation 31 pt, block titles 38 pt bold.
- Body 26 pt (the usual minimum is 24 pt), tables 23 pt, notes and references 20 pt.

## Figures, tables, and logos

- Figure 1 and Tables 2, 4, and 7 come straight from `figures/` and `tables/`. They keep the paper's numbering, so the poster cannot drift from the paper.
- `assets/ncu-logo.png` comes from Wikimedia Commons (`NCULogo.svg`) and is placed without a background.
- `assets/ntcir-logo.tex` holds the NTCIR logo as TikZ `svg.path` data copied from the header of the NTCIR-19 site. Its colors are `ntcirletter` and `ntcirleaf`, which default to the official values. The poster sets the letters to white so they read on the navy band.

## Checks

- One page, A0, and every font embedded (`pdfinfo`, `pdffonts`).
- At 25% zoom, the title and block titles are readable.
