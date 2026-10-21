# Poster brief

## Assumptions and open items

- **Size:** A0, portrait (841 × 1189 mm). The venue requires A0 but does not state an orientation; portrait is assumed and should be confirmed in the Poster Questionnaire (due 2026-10-30 AoE).
- **File name:** `NTCIR19-[TASK]-[FamilyName+Initial]-poster.pdf`, where `[FamilyName+Initial]` is the first author's family name and initial. Not yet decided.
- **Source:** the camera-ready manuscript at `45678XYZ/AI-Cup-2026-ESG-Paper@f91f88b`. Every number on the poster comes from it.
- **Branding:** the NCU logo (`assets/ncu-logo.png`, from Wikimedia Commons `NCULogo.svg`) sits left of the title and the NTCIR logo (`assets/ntcir-logo.tex`, from the NTCIR-19 site) sits right. The palette (navy and leaf green) comes from the NTCIR logo and also suits the ESG topic.
- **Contact:** a QR code to the reproduction capsule (`github.com/tom1030507/ntcir19-esg-validity-layer`) stands in for a contact block. No e-mail address is printed.
- **Style:** style B from the NTCIR-18 comparison (a colored title band and numbered blocks), with the section order of a standard research poster.

## Venue rules (research.nii.ac.jp/ntcir/ntcir-19/poster.html)

- A0 poster, PDF with all fonts embedded.
- Upload through the Poster Upload Form by 2026-11-27 AoE (hard deadline).
- Every active team presents a poster, with or without an oral slot.
- In-person teams bring a printed copy, or ask the organizers to print it through the questionnaire.

## Audience

NTCIR participants and organizers: NLP and IR researchers who know classification metrics but not this task's label hierarchy.

## Main takeaway

Hierarchical projection is a retraining-free validity layer. It removes every invalid ESG tuple and improves whole-tuple accuracy, but not the official field-wise score. Which metric to trust depends on how the outputs are used.

## Content priorities

1. The problem: 17 of 120 label combinations are legal, and independent argmax emits 12.55% invalid tuples.
2. The rule: greedy top-down projection, compared with 17-state decoding, on identical probabilities.
3. The headline contrast: wF1 Δ −0.001 (undetected) against tuple accuracy +0.035 (p_Holm = .001).
4. Replication: tuple accuracy improves in 32/32 external arms, wF1 in 14/32.
5. The adverse result: searching all 17 states costs −0.006 tuple accuracy (p_Holm = .028).
6. Audit use: validity is a precondition, not proof of correctness, and the metric should match how outputs are used.

## Required visuals

- Figure 1 (hierarchy and the seven decision rules) as the hero figure.
- Table 2 (main results), Table 4 (contrasts), and Table 7 (five-corpus replication), all reused unchanged from `tables/`.
