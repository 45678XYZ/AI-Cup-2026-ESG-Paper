# Poster layout

## Canvas

- A0 portrait, 841 × 1189 mm with 20 mm margins, built from `article` with `tcolorbox` blocks.
- Two-column rows use `tcbraster` with `raster equal height=rows`, so both blocks in a row end on the same line. Full-width rows hold Figure 1 and the result tables.
- Do not leave blank lines between blocks inside a `tcbraster`: a blank line starts a new row, and the blocks then stack instead of sitting side by side.

## Grid (top to bottom)

| Row | Left | Right |
|---|---|---|
| Header band (full width) | NCU logo · title, authors, affiliation, team, venue · NTCIR logo | — |
| A | **1 · Introduction**, ending in a Question callout | **2 · Task & Data** |
| B (full width) | **3 · Methodology**: Figure 1 (50%) and two lists, output rules and evaluation (46%) | — |
| C (full width) | **4 · Results: Chinese Anchor**: Table 2 (57.5%) and Table 4 (39.5%), then two bullets | — |
| D | **5 · Multilingual Replication**: Table 7 and two bullets | **6 · Why the Metrics Disagree** |
| E | **7 · Implications for Audit Use** | **8 · Conclusion**, ending in a takeaway callout |
| Footer (full width) | References [1]–[4] in two columns · QR code to the reproduction capsule | — |

## Color

Taken from the NTCIR logo and darkened for print:

| Token | RGB | Use |
|---|---|---|
| navy | 10, 46, 92 | Header band, block title bars, block frames, sub-headings |
| leaf | 11, 138, 67 | Highlighted numbers, bullet markers, callout edge |
| leaf, 55% tint | — | Section numbers and separators on navy |
| mint | 232, 244, 236 | Callouts and footer background |
| ink | 31, 42, 51 | Body text |
| muted | 90, 103, 114 | Table and figure notes |

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
