# Poster layout

## Canvas

- A0 portrait, 841 × 1189 mm, built with `tikzposter` at 25 pt body text.
- Two columns of equal width below a full-width title strip and a full-width takeaway banner.
- One accent color (deep teal) for block titles and highlighted numbers, dark gray body text, and a white background.

## Panel plan (top to bottom)

| Row | Left column | Right column |
|---|---|---|
| Title strip (full width) | Title, team and venue line, authors, affiliation | — |
| Takeaway banner (full width) | Four headline numbers: 12.55% → 0, wF1 Δ −0.001, tuple acc. +0.035, 32/32 vs 14/32 | — |
| 1 | **1 · Problem**: hierarchy rules, 17/120, 44.3% | **4 · Results: Chinese anchor**: Table 2 and Table 4 |
| 2 | **2 · Method**: Figure 1 (hero) and the two output rules | **5 · Replication**: Table 7 and arm counts |
| 3 | **3 · Setup** | **6 · Why the metrics disagree** |
| 4 | — | **7 · For audit use**: metric-by-use list |
| Footer (full width) | Takeaway sentence and QR code to the reproduction capsule | — |

## Figures and tables

- **Hero:** Figure 1 (`figures/figure1_hierarchy.pdf`), full width of the left column, in the Method block where the seven rules are introduced.
- **Tables:** `tables/table2_main.tex`, `tables/table4_contrasts.tex`, and `tables/table7_multilingual_mechanism.tex`, input unchanged and scaled to the column width. The generated files are reused so the poster cannot drift from the paper.
- Captions stand alone and repeat the unit (three-seed means, document-disjoint).

## Hierarchy

1. Title (largest)
2. Takeaway banner numbers (accent color, bold)
3. Block titles
4. Figure 1 and the result tables
5. Bullets

## Notes

- Keep the body copy near 500 words; the copy deck is the source of truth.
- No logos were supplied, so the title strip carries none.
- Check legibility at A0 by viewing the PDF at 25% zoom, which is roughly how it reads from 1.5 m.
