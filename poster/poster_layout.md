# Poster layout

## Canvas

- A0 portrait, 841 × 1189 mm, built with `tikzposter` at 25 pt body text.
- Two columns of equal width below a full-width title strip.
- One accent color (deep teal) for block titles and highlighted numbers, dark gray body text, and a white background.

## Panel plan (top to bottom)

| Row | Left column | Right column |
|---|---|---|
| Title strip (full width) | NCU logo (left, no background), title, authors, affiliation, team and venue | — |
| Upper | **1 · Problem**: hierarchy rules, 17/120, 12.55%, 44.3%; then **3 · Setup** | **2 · Method**: Figure 1 (hero) and the output rules |
| Middle (full width) | **4 · Results: Chinese anchor**: Table 2 and Table 4 side by side, then two bullets | — |
| Lower | **5 · Replication**: Table 7 and arm counts; then **Takeaway** with the QR code to the reproduction capsule | **6 · Why the metrics disagree**; then **7 · For audit use** with the metric-by-use list |

## Figures and tables

- **Hero:** Figure 1 (`figures/figure1_hierarchy.pdf`), full width of the right column, in the Method block where the seven rules are introduced.
- **Tables:** `tables/table2_main.tex`, `tables/table4_contrasts.tex`, and `tables/table7_multilingual_mechanism.tex`, input unchanged at body size and shrunk only if wider than their box. The generated files are reused so the poster cannot drift from the paper.
- Captions keep the paper's numbering (Figure 1, Tables 2, 4, 7) and repeat the unit (three-seed means, document-disjoint).

## Hierarchy

1. Title (largest)
2. Block titles
3. Figure 1 and the result tables
4. Highlighted numbers in the bullets (accent color, bold)
5. Bullets

## Notes

- Keep the body copy near 500 words; the copy deck is the source of truth.
- The logo sits on the title strip's teal background without a white box.
- Check legibility at A0 by viewing the PDF at 25% zoom, which is roughly how it reads from 1.5 m.
