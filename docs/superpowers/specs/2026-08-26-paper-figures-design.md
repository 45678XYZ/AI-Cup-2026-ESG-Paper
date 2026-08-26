# Paper Figures Design

## Scope

Improve the focused NTCIR manuscript's visual explanation without changing its
methods, experimental arms, numerical results, or inferential claims. Work in
the existing `paper/ntcir19-manuscript` workspace and keep the compiled
`manuscript/build/main.pdf` tracked.

## Figure 1: task and decision architecture

Replace the monochrome technical treatment with a full-width, colour-guided
vector diagram. The left panel explains the four-field hierarchy and the
17-of-120 legal-state constraint. The right panel starts from shared base
probabilities and presents independent argmax, hierarchy projection, and
17-state decoding as alternative routes. Calibration choices remain visually
separate from output rules, and M0--M6 labels remain exact. Red denotes a route
that may emit an invalid tuple; blue and teal denote validity-preserving routes.

The source stays in TikZ. All label-space counts continue to come from
`paper.labels` through generated TeX macros. The rendered PDF remains vector,
reproducible, and font-embedded.

## Figure 2: multilingual repair ledger and metric effects

Replace the multilingual table float in the manuscript body with a compact,
full-width vector figure. Each of the five language rows shows:

- the M0 invalid-tuple percentage;
- projection's pooled N/A and substantive-class repair nets; and
- the M1--M0 weighted-macro-F1 and whole-tuple-accuracy deltas.

The left side uses signed ledger cards rather than length-scaled bars because
the corpora contain different row counts. The right side uses a common delta
axis so the five positive tuple effects and mixed weighted-F1 effects are
visually comparable. Exact labels remain printed in the figure.

`tables/table6_multilingual.tex` remains the canonical generated numeric
artifact and retains its provenance manifest. Figure 2 parses that file and
generates TeX definitions before rendering, so the graphic cannot silently
diverge from the table. The table remains committed but is no longer included
as a manuscript float.

## Visual and publication constraints

- Use colour-blind-friendly blue, teal, vermilion, and neutral grey.
- Preserve meaning in greyscale through shapes, signs, and text labels.
- Use no raster images, copied artwork, decorative model icons, or M7 content.
- Keep the manuscript at no more than eight pages and reject overfull boxes
  wider than the repository's existing threshold.
- Update manuscript checks and the run manifest so both committed figure PDFs
  are required, tracked, and reproducible.
