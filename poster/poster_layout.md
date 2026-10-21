# Poster layout

## Canvas

- A0 portrait, 841 × 1189 mm with 20 mm margins, built from `article` with `tcolorbox` blocks.
- The page has a pale blue-grey ground; blocks are white with 4 mm rounded corners and no frame, so they read as cards without outlines. The header and footer bands use the same radius.
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
| Footer band (full width, navy) | References [1]–[4], one per line · "Code & reproduction" with the repo URL · clickable QR code on a white square | — |

## Color

Navy carries the structure. The two highlight colors carry meaning: **green marks validity** and **blue marks performance**, which mirrors the paper's point that legality and score are separate questions. The navy footer balances the navy header.

| Token | RGB | Use |
|---|---|---|
| navy | 12, 45, 87 | Header and footer bands, block title bars, bullets, sub-headings |
| accent | 26, 99, 176 | Performance results only: +0.035, −0.006, 32/32, 14/32 |
| leaf | 0, 160, 64 | Validity results (17 of 120, Always legal, 12.55% → 0) in bold; section-number badges; short rules under the title, before the two Methodology sub-headings, and before the research question and takeaway; header separators |
| numeral | 150, 184, 224 | Reference numbers on the footer band |
| pagebg | 241, 245, 249 | Page ground behind the white blocks |
| rowtint | 236, 242, 249 | Alternate table rows |
| ink | 28, 36, 46 | Body text |
| muted | 88, 100, 116 | Table and figure notes |

Green text is always bold and large: #00A040 on white is about 3.4:1, enough for bold large text (3:1) but not for small body text (4.5:1). There are no callout boxes and no colored left borders: a rounded card with a colored left edge is the most common mark of a generated layout.

## Type

- Fira Sans throughout, set ragged-right with hyphenation off.
- Title 76 pt bold, authors 40 pt, affiliation 31 pt, block titles 38 pt bold.
- Body 26 pt (the usual minimum is 24 pt), tables 23 pt, notes and references 20 pt.

## Figures, tables, and logos

- Figure 1 and Tables 2, 4, and 7 come straight from `figures/` and `tables/`. They keep the paper's numbering, so the poster cannot drift from the paper.
- `assets/ncu-logo.png` comes from Wikimedia Commons (`NCULogo.svg`) and is placed without a background.
- `assets/ntcir-logo.tex` holds the NTCIR logo as TikZ `svg.path` data copied from the header of the NTCIR-19 site. Its colors are `ntcirletter` and `ntcirleaf`, which default to the official values. The poster sets the letters to white so they read on the navy band.

## Checks

- Links: under XeTeX a link drawn only around a picture gets no clickable area, so the QR code sits inside `\XeTeXLinkBox`.

- One page, A0, and every font embedded (`pdfinfo`, `pdffonts`).
- At 25% zoom, the title and block titles are readable.
